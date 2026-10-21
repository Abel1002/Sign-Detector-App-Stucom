"""
Entrenamiento educativo con modelos clásicos + 3 redes neuronales MLP:

Modelos clásicos:
1) Árbol de Decisión
2) Random Forest

Redes neuronales (3 en total):
3) MLP básica (baseline)
4) MLP tratada - Solución 1 (Global Max Pooling previo)
5) MLP tratada - Solución 3 (Atención espacial dinámica previa)

Objetivo docente:
- Comparar por qué el desplazamiento del dígito afecta al modelo.
- Introducir ideas de pooling y atención sin complicar el stack de despliegue.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
DEFAULT_DATASET_SLUG = "karnikakapoor/digits"
DEFAULT_MODEL_PATH = Path("modelo.joblib")
EXPECTED_CLASSES = list(range(10))


def parse_args() -> argparse.Namespace:
    """Lee configuración de entrenamiento desde CLI."""
    parser = argparse.ArgumentParser(
        description="Entrena modelos clásicos y 3 MLP para dígitos escritos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Ruta local con carpetas 0..9. Si no se indica, se descarga de Kaggle.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=DEFAULT_DATASET_SLUG,
        help=f"Slug de Kaggle para descarga automática. Por defecto: {DEFAULT_DATASET_SLUG}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta de salida del artefacto joblib.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Resolución de trabajo (ejemplo: 28 para 28x28).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Límite por clase (0 = todas). Útil para pruebas rápidas.",
    )
    return parser.parse_args()


def infer_digit_label(image_path: Path) -> int | None:
    """Detecta etiqueta numérica leyendo el nombre de carpeta ancestro (0..9)."""
    for part in reversed(image_path.parts):
        if part.isdigit() and len(part) == 1:
            return int(part)
    return None


def collect_labeled_images(dataset_root: Path, max_images_per_class: int) -> list[tuple[Path, int]]:
    """Recorre dataset y crea pares (ruta, etiqueta)."""
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
    """Reduce resolución + normaliza + flatten para entrada tabular."""
    with Image.open(image_path) as img:
        grayscale = ImageOps.grayscale(img)
        resized = grayscale.resize((image_size, image_size))
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return image_array.reshape(-1)


def load_dataset(dataset_root: Path, image_size: int, max_images_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Carga X e y con validación de clases completas."""
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"No existe la ruta del dataset: {dataset_root}\n"
            "Crea la carpeta y añade subcarpetas 0..9 con imágenes."
        )

    labeled_images = collect_labeled_images(dataset_root, max_images_per_class=max_images_per_class)
    if not labeled_images:
        raise FileNotFoundError(
            "No se encontraron imágenes etiquetadas. "
            "Verifica carpetas 0..9 y extensiones válidas."
        )

    labels_found = sorted({label for _, label in labeled_images})
    missing_labels = [label for label in EXPECTED_CLASSES if label not in labels_found]
    if missing_labels:
        raise ValueError(
            "Faltan clases obligatorias en el dataset: "
            f"{missing_labels}. Deben existir carpetas 0..9 con muestras."
        )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    for image_path, label in labeled_images:
        feature_rows.append(preprocess_image_to_vector(image_path=image_path, image_size=image_size))
        labels.append(label)

    X = np.vstack(feature_rows)
    y = np.asarray(labels, dtype=np.int32)
    return X, y


def resolve_dataset_root(dataset_root: Path | None, dataset_slug: str) -> Path:
    """Usa dataset local o descarga con kagglehub."""
    if dataset_root is not None:
        if not dataset_root.exists():
            raise FileNotFoundError(f"No existe la ruta local indicada: {dataset_root}")
        print(f"[2/8] Usando dataset local: {dataset_root.resolve()}")
        return dataset_root

    print(f"[2/8] Descargando dataset desde Kaggle Hub: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Se descargó el dataset, pero la ruta no existe: {downloaded_path}")
    print(f"Dataset descargado en: {downloaded_path}")
    return downloaded_path


def build_global_maxpool_features(X: np.ndarray, image_size: int) -> np.ndarray:
    """
    Solución 1:
    Sustituye el efecto de promedio global por una reducción tipo máximo espacial.

    Idea:
    - Reorganizar vector -> imagen 2D.
    - Aplicar max-pooling 2x2.
    - Flatten de la salida comprimida.
    """
    if image_size < 2:
        return X

    # Si la resolución es impar, recortamos 1 píxel para poder agrupar en bloques 2x2.
    even_size = image_size if image_size % 2 == 0 else image_size - 1
    X_images = X.reshape(-1, image_size, image_size)[:, :even_size, :even_size]

    pooled = X_images.reshape(
        X_images.shape[0],
        even_size // 2,
        2,
        even_size // 2,
        2,
    ).max(axis=(2, 4))

    return pooled.reshape(X.shape[0], -1)


def build_attention_features(X: np.ndarray, image_size: int) -> np.ndarray:
    """
    Solución 3:
    Atención espacial dinámica basada en intensidad del trazo.

    Idea:
    - Generar máscara por imagen con mayor peso en zonas de trazo.
    - Multiplicar imagen * máscara.
    - Flatten para clasificador.
    """
    X_images = X.reshape(-1, image_size, image_size)
    X_weighted = np.empty_like(X_images, dtype=np.float32)

    for index, image in enumerate(X_images):
        # Normalización local por imagen para detectar zonas más informativas.
        min_value = float(image.min())
        max_value = float(image.max())
        dynamic_range = max(max_value - min_value, 1e-6)
        normalized = (image - min_value) / dynamic_range

        # Máscara en [0.25, 1.0]: nunca anula completamente todo, pero privilegia el trazo.
        attention_mask = 0.25 + 0.75 * normalized
        X_weighted[index] = image * attention_mask

    return X_weighted.reshape(X.shape[0], -1)


def build_models() -> dict[str, Any]:
    """
    Crea todos los modelos.

    Nota sobre redes neuronales MLP:
    - `solver='adam'` se usa por estabilidad y bajo coste de ajuste manual.
    - Alternativas:
      * `sgd`: útil con schedule de learning rate y momentum.
      * `lbfgs`: útil en datasets pequeños, pero menos escalable.
    """
    return {
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",  # Medida de impureza en cada split.
            max_depth=18,  # Control de complejidad para limitar sobreajuste.
            min_samples_leaf=2,  # Evita hojas con una sola muestra.
            random_state=RANDOM_STATE,  # Reproducibilidad.
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=140,  # Número de árboles del ensamble.
            criterion="gini",  # Criterio de split.
            max_depth=22,  # Profundidad máxima por árbol.
            max_features="sqrt",  # Feature bagging por nodo.
            min_samples_leaf=1,  # Hojas más finas para capturar detalle.
            bootstrap=True,  # Muestreo con reemplazo.
            oob_score=True,  # Validación interna OOB.
            n_jobs=-1,  # Paraleliza en todos los núcleos.
            random_state=RANDOM_STATE,  # Reproducibilidad.
        ),
        "mlp_basic": MLPClassifier(
            hidden_layer_sizes=(128,),  # Una capa oculta sencilla (baseline).
            activation="relu",  # ReLU: no linealidad robusta para MLP.
            solver="adam",  # Adam: optimización adaptativa estable.
            alpha=1e-4,  # L2 suave para regularizar.
            learning_rate_init=1e-3,  # LR inicial típica para Adam.
            max_iter=60,  # Límite de iteraciones.
            early_stopping=True,  # Corta al dejar de mejorar validación.
            n_iter_no_change=8,  # Paciencia antes de parar.
            random_state=RANDOM_STATE,  # Reproducibilidad.
        ),
        "mlp_sol1_maxpool": MLPClassifier(
            hidden_layer_sizes=(128,),  # Mismo tamaño para comparar justa solución 1.
            activation="relu",  # Activación no lineal estándar.
            solver="adam",  # Misma familia de optimizador para comparabilidad.
            alpha=1e-4,  # Regularización L2.
            learning_rate_init=1e-3,  # Misma LR base que modelo básico.
            max_iter=70,  # Algo más de margen por cambio de representación.
            early_stopping=True,  # Evita sobreentrenamiento.
            n_iter_no_change=10,  # Paciencia ligeramente superior.
            random_state=RANDOM_STATE,  # Reproducibilidad.
        ),
        "mlp_sol3_attention": MLPClassifier(
            hidden_layer_sizes=(256, 128),  # Mayor capacidad para explotar mapa ponderado.
            activation="relu",  # ReLU.
            solver="adam",  # Adam por convergencia estable.
            alpha=5e-4,  # L2 más fuerte para controlar sobreajuste.
            learning_rate_init=8e-4,  # LR un poco menor por red más profunda.
            max_iter=90,  # Más iteraciones para converger.
            early_stopping=True,  # Parada temprana.
            n_iter_no_change=10,  # Paciencia.
            random_state=RANDOM_STATE,  # Reproducibilidad.
        ),
    }


def train_and_evaluate_models(
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Entrena cada modelo con su representación de entrada correspondiente."""
    models = build_models()
    metrics: dict[str, dict[str, float]] = {}

    feature_key_by_model = {
        "decision_tree": "base",
        "random_forest": "base",
        "mlp_basic": "base",
        "mlp_sol1_maxpool": "maxpool",
        "mlp_sol3_attention": "attention",
    }

    for model_name, model in models.items():
        feature_key = feature_key_by_model[model_name]
        X_train_variant, X_test_variant = feature_sets[feature_key]

        print(f"\n[6/8] Entrenando modelo: {model_name} (features={feature_key})")
        model.fit(X_train_variant, y_train)
        predictions = model.predict(X_test_variant)
        accuracy = accuracy_score(y_test, predictions)

        metrics[model_name] = {
            "accuracy": float(accuracy),
            "feature_variant": feature_key,
        }
        print(f"Accuracy {model_name}: {accuracy:.4f}")

        if model_name in {"random_forest", "mlp_sol3_attention"}:
            print(f"\nReporte de clasificación ({model_name}):")
            print(classification_report(y_test, predictions, digits=4))

    random_forest = models["random_forest"]
    metrics["random_forest"]["oob_score"] = float(random_forest.oob_score_)
    print(f"OOB Score random_forest: {random_forest.oob_score_:.4f}")

    return models, metrics


def main() -> None:
    """Orquesta carga, preprocesado, entrenamiento, evaluación y guardado."""
    args = parse_args()

    print("[1/8] Validando parámetros...")
    if args.image_size <= 0:
        raise ValueError("--image-size debe ser mayor que 0.")
    if args.max_images_per_class < 0:
        raise ValueError("--max-images-per-class no puede ser negativo.")

    dataset_root = resolve_dataset_root(dataset_root=args.dataset_root, dataset_slug=args.dataset_slug)

    print("[3/8] Cargando y preprocesando imágenes...")
    X_base, y = load_dataset(
        dataset_root=dataset_root,
        image_size=args.image_size,
        max_images_per_class=args.max_images_per_class,
    )

    classes, class_counts = np.unique(y, return_counts=True)
    class_distribution = {int(label): int(count) for label, count in zip(classes, class_counts)}
    print(f"Muestras totales: {len(y)}")
    print(f"Distribución por clase: {class_distribution}")
    print(f"Features base por imagen: {X_base.shape[1]}")

    print("[4/8] Generando features tratados para redes neuronales...")
    X_maxpool = build_global_maxpool_features(X_base, image_size=args.image_size)
    X_attention = build_attention_features(X_base, image_size=args.image_size)
    print(f"Features maxpool por imagen: {X_maxpool.shape[1]}")
    print(f"Features attention por imagen: {X_attention.shape[1]}")

    print("[5/8] Split train/test estratificado...")
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=0.2,  # 20% de muestras para evaluación.
        random_state=RANDOM_STATE,  # Reproducibilidad del split.
        stratify=y,  # Mantiene balance por clase.
    )

    feature_sets = {
        "base": (X_base[idx_train], X_base[idx_test]),
        "maxpool": (X_maxpool[idx_train], X_maxpool[idx_test]),
        "attention": (X_attention[idx_train], X_attention[idx_test]),
    }

    models, metrics = train_and_evaluate_models(feature_sets=feature_sets, y_train=y_train, y_test=y_test)

    model_artifact = {
        "decision_tree": models["decision_tree"],
        "random_forest": models["random_forest"],
        "mlp_basic": models["mlp_basic"],
        "mlp_sol1_maxpool": models["mlp_sol1_maxpool"],
        "mlp_sol3_attention": models["mlp_sol3_attention"],
        "metadata": {
            "dataset_type": "handwritten-digits-folder",
            "dataset_source": args.dataset_slug if args.dataset_root is None else "local-path",
            "dataset_root": str(dataset_root.resolve()),
            "image_size": int(args.image_size),
            "feature_count_base": int(X_base.shape[1]),
            "feature_count_maxpool": int(X_maxpool.shape[1]),
            "feature_count_attention": int(X_attention.shape[1]),
            "class_labels": [int(x) for x in classes.tolist()],
            "class_distribution": class_distribution,
            "metrics": metrics,
        },
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        model_artifact,
        args.model_path,
        compress=3,  # Reduce tamaño del artefacto para despliegue.
    )
    print(f"\n[8/8] Modelo guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
