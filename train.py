"""
Entrenamiento educativo de 4 modelos para dígitos escritos (0..9):
1) Árbol de Decisión
2) Random Forest
3) Red Neuronal MLP pequeña
4) Red Neuronal MLP profunda

Objetivo docente:
- Mantener el mismo dataset y preprocesado en los 4 modelos.
- Comparar resultados con una base metodológica justa.
- Entender qué cambia al pasar de árboles a redes neuronales.
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

# Extensiones de imagen que se aceptan al escanear el dataset.
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Semilla global para reproducibilidad de resultados.
RANDOM_STATE = 42

# Dataset por defecto según requisito del proyecto.
DEFAULT_DATASET_SLUG = "karnikakapoor/digits"

# Artefacto final que consumirá la app de Streamlit.
DEFAULT_MODEL_PATH = Path("modelo.joblib")

# Clases esperadas (problema multiclase 0..9).
EXPECTED_CLASSES = list(range(10))


def parse_args() -> argparse.Namespace:
    """Lee argumentos de terminal para controlar datos, tamaño y salida."""
    parser = argparse.ArgumentParser(
        description="Entrena 4 modelos (DT, RF, MLP1, MLP2) para clasificación de dígitos escritos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Ruta local con carpetas 0..9. "
            "Si no se indica, se descarga desde Kaggle."
        ),
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
        help="Ruta donde se guardará modelo.joblib.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Resolución cuadrada de trabajo (ej. 28 = 28x28).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Límite por clase (0 = todas). Útil para pruebas rápidas.",
    )
    return parser.parse_args()


def infer_digit_label(image_path: Path) -> int | None:
    """Extrae la etiqueta 0..9 desde la carpeta ancestro de la imagen."""
    for part in reversed(image_path.parts):
        if part.isdigit() and len(part) == 1:
            return int(part)
    return None


def collect_labeled_images(dataset_root: Path, max_images_per_class: int) -> list[tuple[Path, int]]:
    """Recorre dataset y construye pares (ruta_imagen, etiqueta)."""
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
    Convierte imagen a vector numérico:
    1) Escala de grises.
    2) Reducción de resolución.
    3) Normalización [0, 1].
    4) Aplanado 1D.
    """
    with Image.open(image_path) as img:
        grayscale = ImageOps.grayscale(img)
        resized = grayscale.resize((image_size, image_size))
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return image_array.reshape(-1)


def load_dataset(dataset_root: Path, image_size: int, max_images_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Carga X e y desde carpetas 0..9, con validaciones de integridad."""
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
    """Decide si usar dataset local o descargarlo de Kaggle."""
    if dataset_root is not None:
        if not dataset_root.exists():
            raise FileNotFoundError(f"No existe la ruta local indicada: {dataset_root}")
        print(f"[2/7] Usando dataset local: {dataset_root.resolve()}")
        return dataset_root

    print(f"[2/7] Descargando dataset desde Kaggle Hub: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Se descargó el dataset, pero la ruta no existe: {downloaded_path}")
    print(f"Dataset descargado en: {downloaded_path}")
    return downloaded_path


def build_models() -> dict[str, Any]:
    """
    Construye los 4 modelos.

    Comentario pedagógico sobre MLP y `solver='adam'`:
    - Adam funciona bien cuando hay muchos parámetros y datos ruidosos.
    - Converge más estable que SGD puro sin requerir mucha calibración manual.
    - Alternativas útiles en casos parecidos:
      * solver='sgd' (control fino con momentum, útil para experimentar LR schedules).
      * solver='lbfgs' (puede ir bien en datasets pequeños, pero escala peor en datos grandes).

    Si el problema fuera texto en vez de imágenes:
    - En clásico: Logistic Regression, Linear SVM, Naive Bayes sobre TF-IDF.
    - En deep learning: LSTM/GRU o Transformers (BERT-like) con optimizadores AdamW.
    """
    return {
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",  # Criterio de impureza de Gini para decidir cada split del árbol.
            max_depth=18,  # Límite de profundidad para evitar sobreajuste extremo.
            min_samples_leaf=2,  # Mínimo de muestras en hoja para suavizar decisiones finales.
            random_state=RANDOM_STATE,  # Semilla reproducible para construir el árbol.
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=140,  # Número de árboles en el bosque (más árboles = más estabilidad).
            criterion="gini",  # Mismo criterio de split que el árbol individual.
            max_depth=22,  # Profundidad máxima por árbol, algo más alta por robustez del ensamble.
            max_features="sqrt",  # Feature bagging: usa raíz(n_features) por split.
            min_samples_leaf=1,  # Permite hojas más específicas en cada árbol del bosque.
            bootstrap=True,  # Activa muestreo con reemplazo para cada árbol.
            oob_score=True,  # Calcula OOB score para validación interna del ensamble.
            n_jobs=-1,  # Paraleliza usando todos los núcleos disponibles.
            random_state=RANDOM_STATE,  # Semilla reproducible para bootstrapping y splits.
        ),
        "mlp_small": MLPClassifier(
            hidden_layer_sizes=(128,),  # Una capa oculta de 128 neuronas (red simple).
            activation="relu",  # ReLU: rápida y estándar para redes densas modernas.
            solver="adam",  # Adam: optimizador adaptativo estable para este tipo de problema.
            alpha=1e-4,  # Regularización L2 suave para reducir sobreajuste.
            learning_rate_init=1e-3,  # Tasa inicial de aprendizaje de Adam.
            max_iter=60,  # Máximo de épocas/iteraciones del optimizador.
            early_stopping=True,  # Corta entrenamiento si no mejora validación.
            n_iter_no_change=8,  # Paciencia antes de detener por falta de mejora.
            random_state=RANDOM_STATE,  # Semilla reproducible para pesos iniciales y split interno.
        ),
        "mlp_deep": MLPClassifier(
            hidden_layer_sizes=(256, 128),  # Dos capas ocultas: más capacidad no lineal.
            activation="relu",  # ReLU mantiene gradientes útiles y entreno eficiente.
            solver="adam",  # Adam también aquí por robustez en convergencia.
            alpha=5e-4,  # L2 más fuerte para compensar mayor capacidad de la red profunda.
            learning_rate_init=8e-4,  # LR algo menor para entreno más estable en red más grande.
            max_iter=80,  # Más iteraciones para permitir convergencia de arquitectura más profunda.
            early_stopping=True,  # Evita sobreentrenar cuando ya no mejora.
            n_iter_no_change=10,  # Paciencia mayor por curva de aprendizaje más lenta.
            random_state=RANDOM_STATE,  # Reproducibilidad.
        ),
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Entrena los 4 modelos y calcula métricas comparables."""
    models = build_models()
    metrics: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        print(f"\n[5/7] Entrenando modelo: {model_name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        metrics[model_name] = {"accuracy": float(accuracy)}
        print(f"Accuracy {model_name}: {accuracy:.4f}")

        if model_name in {"random_forest", "mlp_deep"}:
            print(f"\nReporte de clasificación ({model_name}):")
            print(classification_report(y_test, predictions, digits=4))

    random_forest = models["random_forest"]
    metrics["random_forest"]["oob_score"] = float(random_forest.oob_score_)
    print(f"OOB Score random_forest: {random_forest.oob_score_:.4f}")

    return models, metrics


def main() -> None:
    """Orquesta todo el pipeline de entrenamiento y guardado."""
    args = parse_args()

    print("[1/7] Validando parámetros...")
    if args.image_size <= 0:
        raise ValueError("--image-size debe ser mayor que 0.")
    if args.max_images_per_class < 0:
        raise ValueError("--max-images-per-class no puede ser negativo.")

    dataset_root = resolve_dataset_root(dataset_root=args.dataset_root, dataset_slug=args.dataset_slug)

    print("[3/7] Cargando y preprocesando imágenes...")
    X, y = load_dataset(
        dataset_root=dataset_root,
        image_size=args.image_size,
        max_images_per_class=args.max_images_per_class,
    )

    classes, class_counts = np.unique(y, return_counts=True)
    class_distribution = {int(label): int(count) for label, count in zip(classes, class_counts)}
    print(f"Muestras totales: {len(y)}")
    print(f"Distribución por clase: {class_distribution}")
    print(f"Features por imagen: {X.shape[1]}")

    print("[4/7] Split train/test estratificado...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 20% test para medir generalización en datos no vistos.
        random_state=RANDOM_STATE,  # Reproducibilidad del split.
        stratify=y,  # Mantiene proporción de clases en train y test.
    )

    models, metrics = train_and_evaluate_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    model_artifact = {
        "decision_tree": models["decision_tree"],
        "random_forest": models["random_forest"],
        "mlp_small": models["mlp_small"],
        "mlp_deep": models["mlp_deep"],
        "metadata": {
            "dataset_type": "handwritten-digits-folder",
            "dataset_source": args.dataset_slug if args.dataset_root is None else "local-path",
            "dataset_root": str(dataset_root.resolve()),
            "image_size": int(args.image_size),
            "feature_count": int(X.shape[1]),
            "class_labels": [int(x) for x in classes.tolist()],
            "class_distribution": class_distribution,
            "metrics": metrics,
        },
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        model_artifact,
        args.model_path,
        compress=3,  # Compresión media: reduce tamaño sin ralentizar demasiado guardado/carga.
    )
    print(f"\n[7/7] Modelo guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
