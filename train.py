"""
Entrenamiento educativo para dígitos 0..9 con:
- Modelos clásicos: Árbol y Random Forest.
- Redes neuronales:
  1) MLP básica.
  2) CNN (MobileNetV2) + Global Max Pooling.
  3) MLP con atención espacial dinámica + recentrado.

La variante CNN se aplica según teoría correcta:
- El pooling global se hace sobre mapas de características convolucionales,
  no sobre píxeles crudos aplanados.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
DEFAULT_DATASET_SLUG = "karnikakapoor/digits"
DEFAULT_MODEL_PATH = Path("modelo.joblib")
EXPECTED_CLASSES = list(range(10))
MLP_ATTENTION_KEY = "mlp_attention_recentered"
CNN_GMP_KEY = "cnn_global_max_pooling"
LEGACY_MLP_ATTENTION_KEY = "mlp_sol3_attention"
LEGACY_CNN_GMP_KEY = "cnn_sol1_global_maxpool"


def parse_args() -> argparse.Namespace:
    """
    Define la interfaz CLI del entrenamiento.

    Base metodológica:
    - Mantiene parámetros clásicos de ingeniería de datos (fuente, tamaño, salida).
    - Expone hiperparámetros para comparar modelos tabulares y CNN bajo el mismo pipeline.

    Aporte al entrenamiento:
    - Permite reproducibilidad y pruebas controladas (mismo script, distintos perfiles).
    - Incluye defaults optimizados para CPU en aula.
    """
    parser = argparse.ArgumentParser(
        description="Entrena clásicos + 3 redes (MLP básica, CNN+GMP, MLP con atención)."
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Ruta local de dataset 0..9.")
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=DEFAULT_DATASET_SLUG,
        help=f"Slug Kaggle para descarga automática. Por defecto: {DEFAULT_DATASET_SLUG}",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Ruta de modelo.joblib.")
    parser.add_argument("--image-size", type=int, default=28, help="Resolución base para modelos tabulares.")
    parser.add_argument("--max-images-per-class", type=int, default=250, help="Límite por clase (0=todas).")
    parser.add_argument("--cnn-image-size", type=int, default=64, help="Resolución para CNN con Global Max Pooling.")
    parser.add_argument("--cnn-epochs", type=int, default=4, help="Épocas de entrenamiento de la CNN.")
    parser.add_argument("--cnn-batch-size", type=int, default=64, help="Batch size para la CNN.")
    parser.add_argument("--cnn-lr", type=float, default=2e-3, help="Learning rate de la cabeza CNN.")
    return parser.parse_args()


def infer_digit_label(image_path: Path) -> int | None:
    """
    Infere la etiqueta (0..9) desde la jerarquía de carpetas del dataset.

    Base metodológica:
    - Sigue el convenio común de datasets en carpetas por clase.

    Aporte al entrenamiento:
    - Convierte estructura de archivos en supervisión (X, y) sin anotaciones extra.
    """
    for part in reversed(image_path.parts):
        if part.isdigit() and len(part) == 1:
            return int(part)
    return None


def collect_labeled_images(dataset_root: Path, max_images_per_class: int) -> list[tuple[Path, int]]:
    """
    Recorre el dataset y reúne muestras válidas por clase.

    Base metodológica:
    - Filtra extensiones soportadas y normaliza el muestreo por etiqueta.
    - Aplica límite opcional por clase para controlar coste/tiempo.

    Aporte al entrenamiento:
    - Genera un conjunto balanceado y consistente para entrenar y comparar modelos.
    """
    images_by_label: dict[int, list[Path]] = defaultdict(list)

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        label = infer_digit_label(file_path)
        if label is None or label not in EXPECTED_CLASSES:
            continue
        images_by_label[label].append(file_path)

    for label, paths in images_by_label.items():
        paths.sort()
        if max_images_per_class > 0:
            images_by_label[label] = paths[:max_images_per_class]

    labeled_images: list[tuple[Path, int]] = []
    for label in EXPECTED_CLASSES:
        labeled_images.extend((image_path, label) for image_path in images_by_label.get(label, []))
    return labeled_images


def preprocess_image_to_vector(image_path: Path, image_size: int) -> np.ndarray:
    """
    Convierte una imagen en vector tabular para modelos de scikit-learn.

    Base metodológica:
    - Escala de grises + resize fijo + normalización [0,1] + flatten.

    Aporte al entrenamiento:
    - Crea una representación homogénea para Árbol, Bosque y MLP tabulares.
    """
    with Image.open(image_path) as img:
        grayscale = ImageOps.grayscale(img)
        resized = grayscale.resize((image_size, image_size))
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return image_array.reshape(-1)


def load_dataset(dataset_root: Path, image_size: int, max_images_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye matrices X/y para modelos tabulares.

    Base metodológica:
    - Valida clases 0..9 presentes.
    - Aplica el mismo preprocesado a todas las muestras.

    Aporte al entrenamiento:
    - Entrega tensores tabulares consistentes para baseline clásico y MLP.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe la ruta del dataset: {dataset_root}")

    labeled_images = collect_labeled_images(dataset_root, max_images_per_class=max_images_per_class)
    if not labeled_images:
        raise FileNotFoundError("No se encontraron imágenes etiquetadas válidas.")

    labels_found = sorted({label for _, label in labeled_images})
    missing_labels = [label for label in EXPECTED_CLASSES if label not in labels_found]
    if missing_labels:
        raise ValueError(f"Faltan clases en dataset: {missing_labels}")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    for image_path, label in labeled_images:
        feature_rows.append(preprocess_image_to_vector(image_path=image_path, image_size=image_size))
        labels.append(label)

    X = np.vstack(feature_rows)
    y = np.asarray(labels, dtype=np.int32)
    return X, y


def resolve_dataset_root(dataset_root: Path | None, dataset_slug: str) -> Path:
    """
    Resuelve la fuente del dataset (local o descarga automática).

    Base metodológica:
    - Prioriza ruta explícita del usuario y usa KaggleHub como fallback reproducible.

    Aporte al entrenamiento:
    - Reduce fricción de preparación y mantiene trazabilidad del origen de datos.
    """
    if dataset_root is not None:
        if not dataset_root.exists():
            raise FileNotFoundError(f"No existe la ruta local indicada: {dataset_root}")
        print(f"[2/9] Usando dataset local: {dataset_root.resolve()}")
        return dataset_root

    print(f"[2/9] Descargando dataset desde Kaggle Hub: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Se descargó dataset, pero no existe ruta: {downloaded_path}")
    print(f"Dataset descargado en: {downloaded_path}")
    return downloaded_path


def build_attention_features(X: np.ndarray, image_size: int) -> np.ndarray:
    """
    Genera features para MLP con recentrado + atención espacial sobre tinta.

    Base metodológica:
    - Detecta bounding box del trazo, recentra en un lienzo canónico y realza píxeles útiles
      con una máscara de atención proporcional a la tinta.

    Aporte al entrenamiento:
    - Reduce sesgo posicional del vector aplanado y mejora robustez en dígitos desplazados.
    """
    X_images = X.reshape(-1, image_size, image_size)
    X_weighted = np.empty_like(X_images, dtype=np.float32)

    for index, image in enumerate(X_images):
        ink = 1.0 - image
        rows, cols = np.where(ink > 0.1)

        if len(rows) > 0:
            y_min, y_max = int(np.min(rows)), int(np.max(rows))
            x_min, x_max = int(np.min(cols)), int(np.max(cols))
            cropped = Image.fromarray((image * 255.0).astype(np.uint8)).crop((x_min, y_min, x_max + 1, y_max + 1))
            width, height = cropped.size
            scale = (image_size * 0.7) / max(width, height, 1)
            new_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            resized = cropped.resize(new_size, Image.Resampling.BILINEAR)

            centered = Image.new("L", (image_size, image_size), color=255)
            offset_x = (image_size - new_size[0]) // 2
            offset_y = (image_size - new_size[1]) // 2
            centered.paste(resized, (offset_x, offset_y))
            image = np.asarray(centered, dtype=np.float32) / 255.0
            ink = 1.0 - image

        attention_mask = 0.25 + 0.75 * ink
        weighted_ink = np.clip(ink * attention_mask, 0.0, 1.0)
        X_weighted[index] = 1.0 - weighted_ink

    return X_weighted.reshape(X.shape[0], -1)


class InMemoryDigitDataset(Dataset):
    """
    Dataset PyTorch que precarga imágenes redimensionadas en RAM.

    Base metodológica:
    - Desacopla lectura/decodificación de disco del loop de entrenamiento.

    Aporte al entrenamiento:
    - Disminuye latencia por época y acelera entrenamiento en CPU.
    """

    def __init__(self, samples: list[tuple[Path, int]], cnn_image_size: int, transform) -> None:
        """
        Materializa en memoria la versión base de cada muestra.

        Aporte:
        - El resize ocurre una sola vez; las transformaciones estocásticas se aplican al vuelo.
        """
        self.transform = transform
        self.data: list[tuple[Image.Image, int]] = []

        for image_path, label in samples:
            with Image.open(image_path) as img:
                rgb_small = img.convert("RGB").resize((cnn_image_size, cnn_image_size), Image.Resampling.BILINEAR)
            self.data.append((rgb_small, label))

    def __len__(self) -> int:
        """Devuelve número de muestras precargadas para DataLoader."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Aplica transformaciones y retorna par (tensor, etiqueta)."""
        image, label = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MobileNetV2GlobalMaxPool(nn.Module):
    """
    Variante CNN:
    - Backbone convolucional MobileNetV2 preentrenado.
    - Global Max Pooling adaptativo sobre mapas convolucionales.
    - Clasificador denso final.
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Inicializa arquitectura CNN para clasificación 0..9.

        Base metodológica:
        - Transfer learning con extractor visual preentrenado.
        - Reducción espacial global para tolerar desplazamiento.
        """
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        """
        Ejecuta inferencia: features -> pooling global -> clasificador.

        Aporte:
        - Colapsa dimensión espacial y conserva señal semántica por canal.
        """
        x = self.features(x)
        x = self.global_max_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class FocalLoss(nn.Module):
    """
    Focal Loss con label smoothing:
    reduce el peso de ejemplos fáciles y concentra gradiente en casos difíciles.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05) -> None:
        """
        Configura penalización focal + suavizado de etiquetas.

        Aporte:
        - Prioriza ejemplos difíciles y evita sobreconfianza extrema.
        """
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula la pérdida focal media del batch."""
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


def build_models() -> dict[str, Any]:
    """
    Define el conjunto de modelos tabulares comparables.

    Base metodológica:
    - Incluye baseline árbol, ensamblado de bosque y MLPs con distinta capacidad.

    Aporte al entrenamiento:
    - Permite comparar rendimiento de enfoques clásicos y MLP en el mismo split.
    """
    return {
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=140,
            criterion="gini",
            max_depth=22,
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "mlp_basic": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=RANDOM_STATE,
        ),
        MLP_ATTENTION_KEY: MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=5e-4,
            learning_rate_init=8e-4,
            max_iter=90,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_evaluate_tabular_models(
    X_base_train: np.ndarray,
    X_base_test: np.ndarray,
    X_attention_train: np.ndarray,
    X_attention_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """
    Entrena y evalúa todos los modelos tabulares en condiciones homogéneas.

    Base metodológica:
    - Mismo split estratificado.
    - Features base para modelos estándar y features con atención para MLP especializada.

    Aporte al entrenamiento:
    - Produce métricas comparables y reportes de diagnóstico por modelo.
    """
    models_dict = build_models()
    metrics: dict[str, dict[str, float]] = {}

    for model_name, model in models_dict.items():
        if model_name == MLP_ATTENTION_KEY:
            X_train_variant = X_attention_train
            X_test_variant = X_attention_test
            feature_variant = "attention"
        else:
            X_train_variant = X_base_train
            X_test_variant = X_base_test
            feature_variant = "base"

        print(f"\n[6/9] Entrenando modelo: {model_name} (features={feature_variant})")
        model.fit(X_train_variant, y_train)
        predictions = model.predict(X_test_variant)
        accuracy = accuracy_score(y_test, predictions)
        metrics[model_name] = {"accuracy": float(accuracy), "feature_variant": feature_variant}
        print(f"Accuracy {model_name}: {accuracy:.4f}")

        if model_name in {"random_forest", MLP_ATTENTION_KEY}:
            print(f"\nReporte de clasificación ({model_name}):")
            print(classification_report(y_test, predictions, digits=4))

    random_forest = models_dict["random_forest"]
    metrics["random_forest"]["oob_score"] = float(random_forest.oob_score_)
    print(f"OOB Score random_forest: {random_forest.oob_score_:.4f}")

    return models_dict, metrics


def train_cnn_global_max_pooling(
    labeled_images: list[tuple[Path, int]],
    cnn_image_size: int,
    cnn_epochs: int,
    cnn_batch_size: int,
    cnn_lr: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Entrena la CNN con Global Max Pooling y fine-tuning eficiente en CPU.

    Base metodológica:
    - Transfer learning (MobileNetV2), descongelado parcial, Focal Loss y AdamW.
    - Augmentation geométrica para robustez posicional.

    Aporte al entrenamiento:
    - Aprende representación visual desplazamiento-tolerante con coste de cómputo acotado.
    """
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[7/9] Entrenando CNN con Global Max Pooling (Modo rápido CPU + Fine-Tuning) en: {device}")

    labels = np.array([label for _, label in labeled_images], dtype=np.int32)
    indices = np.arange(len(labeled_images))
    train_idx, test_idx, _, _ = train_test_split(
        indices,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_samples = [labeled_images[i] for i in train_idx]
    test_samples = [labeled_images[i] for i in test_idx]

    transform_train = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.30, 0.30),
                scale=(0.50, 0.95),
                fill=255,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Precargando muestras CNN en RAM...")
    train_dataset = InMemoryDigitDataset(train_samples, cnn_image_size=cnn_image_size, transform=transform_train)
    test_dataset = InMemoryDigitDataset(test_samples, cnn_image_size=cnn_image_size, transform=transform_eval)
    train_loader = DataLoader(train_dataset, batch_size=cnn_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cnn_batch_size, shuffle=False, num_workers=0)

    model = MobileNetV2GlobalMaxPool(num_classes=10).to(device)

    # Bloques iniciales congelados (features universales).
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Fine-tuning quirúrgico: solo último bloque para reducir coste de backward en CPU.
    for parameter in model.features[18].parameters():
        parameter.requires_grad = True

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features[18].parameters(), "lr": cnn_lr * 0.2},
            {"params": model.classifier.parameters(), "lr": cnn_lr},
        ],
        weight_decay=1e-2,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_test_acc = 0.0

    for epoch in range(cnn_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * targets.size(0)
            running_correct += int((predictions == targets).sum().item())
            running_total += int(targets.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                eval_correct += int((predictions == targets).sum().item())
                eval_total += int(targets.size(0))
        eval_acc = eval_correct / max(eval_total, 1)

        print(
            f"[CNN][Epoch {epoch + 1}/{cnn_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={eval_acc:.4f}"
        )

        if eval_acc >= best_test_acc:
            best_test_acc = eval_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    all_predictions: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_targets.extend(targets.numpy().tolist())

    print(f"\nReporte de clasificación ({CNN_GMP_KEY}):")
    print(classification_report(all_targets, all_predictions, digits=4))

    metrics = {
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "feature_variant": "cnn_global_max_pool_finetuned",
    }

    state_dict_cpu = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    config = {
        "num_classes": 10,
        "cnn_image_size": cnn_image_size,
        "weights": "MobileNet_V2_Weights.DEFAULT",
        "pooling": "AdaptiveMaxPool2d((1,1))",
    }
    return {"state_dict": state_dict_cpu, "config": config}, metrics


def main() -> None:
    """
    Orquesta el pipeline completo: carga, features, entrenamiento, evaluación y guardado.

    Base metodológica:
    - Separa flujo tabular y flujo CNN para comparar ambos paradigmas.

    Aporte al entrenamiento:
    - Produce un artefacto único (`modelo.joblib`) listo para inferencia en app.
    """
    args = parse_args()

    print("[1/9] Validando parámetros...")
    if args.image_size <= 0:
        raise ValueError("--image-size debe ser mayor que 0.")
    if args.max_images_per_class < 0:
        raise ValueError("--max-images-per-class no puede ser negativo.")
    if args.cnn_image_size <= 0:
        raise ValueError("--cnn-image-size debe ser mayor que 0.")
    if args.cnn_epochs <= 0:
        raise ValueError("--cnn-epochs debe ser mayor que 0.")
    if args.cnn_batch_size <= 0:
        raise ValueError("--cnn-batch-size debe ser mayor que 0.")

    dataset_root = resolve_dataset_root(dataset_root=args.dataset_root, dataset_slug=args.dataset_slug)

    print("[3/9] Cargando y preprocesando imágenes base...")
    X_base, y = load_dataset(
        dataset_root=dataset_root,
        image_size=args.image_size,
        max_images_per_class=args.max_images_per_class,
    )
    labeled_images = collect_labeled_images(dataset_root, max_images_per_class=args.max_images_per_class)

    classes, class_counts = np.unique(y, return_counts=True)
    class_distribution = {int(label): int(count) for label, count in zip(classes, class_counts)}
    print(f"Muestras totales: {len(y)}")
    print(f"Distribución por clase: {class_distribution}")
    print(f"Features base por imagen: {X_base.shape[1]}")

    print("[4/9] Generando features MLP de atención espacial + recentrado...")
    X_attention = build_attention_features(X_base, image_size=args.image_size)
    print(f"Features attention por imagen: {X_attention.shape[1]}")

    print("[5/9] Split train/test estratificado para modelos tabulares...")
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tabular_models, tabular_metrics = train_and_evaluate_tabular_models(
        X_base_train=X_base[idx_train],
        X_base_test=X_base[idx_test],
        X_attention_train=X_attention[idx_train],
        X_attention_test=X_attention[idx_test],
        y_train=y_train,
        y_test=y_test,
    )

    cnn_model_bundle, cnn_metrics = train_cnn_global_max_pooling(
        labeled_images=labeled_images,
        cnn_image_size=args.cnn_image_size,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        cnn_lr=args.cnn_lr,
    )

    print(f"Accuracy {CNN_GMP_KEY}: {cnn_metrics['accuracy']:.4f}")

    model_artifact = {
        "decision_tree": tabular_models["decision_tree"],
        "random_forest": tabular_models["random_forest"],
        "mlp_basic": tabular_models["mlp_basic"],
        MLP_ATTENTION_KEY: tabular_models[MLP_ATTENTION_KEY],
        LEGACY_MLP_ATTENTION_KEY: tabular_models[MLP_ATTENTION_KEY],
        f"{CNN_GMP_KEY}_state_dict": cnn_model_bundle["state_dict"],
        f"{CNN_GMP_KEY}_config": cnn_model_bundle["config"],
        f"{LEGACY_CNN_GMP_KEY}_state_dict": cnn_model_bundle["state_dict"],
        f"{LEGACY_CNN_GMP_KEY}_config": cnn_model_bundle["config"],
        "metadata": {
            "dataset_type": "handwritten-digits-folder",
            "dataset_source": args.dataset_slug if args.dataset_root is None else "local-path",
            "dataset_root": str(dataset_root.resolve()),
            "image_size": int(args.image_size),
            "cnn_image_size": int(args.cnn_image_size),
            "feature_count_base": int(X_base.shape[1]),
            "feature_count_attention": int(X_attention.shape[1]),
            "class_labels": [int(x) for x in classes.tolist()],
            "class_distribution": class_distribution,
            "metrics": {
                **tabular_metrics,
                LEGACY_MLP_ATTENTION_KEY: tabular_metrics[MLP_ATTENTION_KEY],
                CNN_GMP_KEY: cnn_metrics,
                LEGACY_CNN_GMP_KEY: cnn_metrics,
            },
        },
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, args.model_path, compress=3)
    print(f"\n[9/9] Modelo guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
