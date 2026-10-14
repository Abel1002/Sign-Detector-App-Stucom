"""
Entrenamiento de clasificación de números escritos a mano (0..9) con:
1) Árbol de Decisión
2) Random Forest

Dataset esperado:
dataset/
├── 0/
├── 1/
...
└── 9/

Cada carpeta contiene imágenes del dígito correspondiente sobre fondo blanco.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import kagglehub
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
DEFAULT_DATASET_SLUG = "karnikakapoor/digits"
DEFAULT_MODEL_PATH = Path("modelo.joblib")
EXPECTED_CLASSES = list(range(10))


def parse_args() -> argparse.Namespace:
    """Configura argumentos para controlar dataset, tamaño de imagen y salida."""
    parser = argparse.ArgumentParser(
        description="Entrena modelos de clasificación para dígitos escritos a mano (0..9)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Ruta raíz local del dataset con carpetas 0..9. "
            "Si se omite, se descargará automáticamente desde Kaggle."
        ),
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=DEFAULT_DATASET_SLUG,
        help=f"Dataset de Kaggle a descargar si no se indica --dataset-root. Por defecto: {DEFAULT_DATASET_SLUG}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta de salida del modelo joblib. Por defecto: ./modelo.joblib",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Tamaño final cuadrado de entrada (ejemplo: 28 para 28x28).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help=(
            "Límite por clase para reducir peso/tiempo (0 = usar todas las imágenes). "
            "Útil para pruebas rápidas."
        ),
    )
    return parser.parse_args()


def infer_digit_label(image_path: Path) -> int | None:
    """Devuelve la etiqueta del dígito según la carpeta ancestro '0'..'9'."""
    for part in reversed(image_path.parts):
        if part.isdigit() and len(part) == 1:
            return int(part)
    return None


def collect_labeled_images(dataset_root: Path, max_images_per_class: int) -> list[tuple[Path, int]]:
    """Encuentra imágenes válidas y aplica muestreo opcional por clase."""
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
    Convierte cada imagen a vector de entrada:
    - Escala de grises
    - Redimensión a image_size x image_size
    - Normalización [0, 1]
    - Aplanado 1D
    """
    with Image.open(image_path) as img:
        grayscale = ImageOps.grayscale(img)
        resized = grayscale.resize((image_size, image_size))
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return image_array.reshape(-1)


def load_dataset(dataset_root: Path, image_size: int, max_images_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Carga y transforma el dataset a matrices X e y para scikit-learn."""
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"No existe la ruta del dataset: {dataset_root}\n"
            "Crea la carpeta y añade subcarpetas 0..9 con imágenes."
        )

    labeled_images = collect_labeled_images(dataset_root, max_images_per_class=max_images_per_class)
    if not labeled_images:
        raise FileNotFoundError(
            "No se encontraron imágenes etiquetadas. "
            "Verifica que existan carpetas 0..9 con imágenes dentro."
        )

    labels_found = sorted({label for _, label in labeled_images})
    missing_labels = [label for label in EXPECTED_CLASSES if label not in labels_found]
    if missing_labels:
        raise ValueError(
            "Faltan clases obligatorias en el dataset: "
            f"{missing_labels}. Deben existir carpetas 0..9 con ejemplos."
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
    """Resuelve dataset local o descarga automática de Kaggle."""
    if dataset_root is not None:
        if not dataset_root.exists():
            raise FileNotFoundError(f"No existe la ruta local indicada: {dataset_root}")
        print(f"[2/6] Usando dataset local: {dataset_root.resolve()}")
        return dataset_root

    print(f"[2/6] Descargando dataset desde Kaggle Hub: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Se descargó el dataset pero la ruta no existe: {downloaded_path}")
    print(f"Dataset descargado en: {downloaded_path}")
    return downloaded_path


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> tuple[DecisionTreeClassifier, RandomForestClassifier]:
    """Entrena Árbol de Decisión y Random Forest con configuración educativa."""
    print("[4/6] Entrenando Árbol de Decisión (Gini)...")
    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    decision_tree.fit(X_train, y_train)

    print("[5/6] Entrenando Random Forest (Bagging + Voto Mayoritario)...")
    random_forest = RandomForestClassifier(
        n_estimators=140,
        criterion="gini",
        max_depth=22,
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    random_forest.fit(X_train, y_train)
    return decision_tree, random_forest


def main() -> None:
    args = parse_args()

    print("[1/6] Validando parámetros...")
    if args.image_size <= 0:
        raise ValueError("--image-size debe ser mayor que 0.")
    if args.max_images_per_class < 0:
        raise ValueError("--max-images-per-class no puede ser negativo.")

    dataset_root = resolve_dataset_root(dataset_root=args.dataset_root, dataset_slug=args.dataset_slug)

    print("[3/6] Cargando y preprocesando imágenes...")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    decision_tree, random_forest = train_models(X_train, y_train)

    dt_predictions = decision_tree.predict(X_test)
    rf_predictions = random_forest.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    print("\n=== Métricas ===")
    print(f"Accuracy Árbol de Decisión: {dt_accuracy:.4f}")
    print(f"Accuracy Random Forest   : {rf_accuracy:.4f}")
    print(f"OOB Score Random Forest  : {random_forest.oob_score_:.4f}")
    print("\nReporte de clasificación (Random Forest):")
    print(classification_report(y_test, rf_predictions, digits=4))

    model_artifact = {
        "decision_tree": decision_tree,
        "random_forest": random_forest,
        "metadata": {
            "dataset_type": "handwritten-digits-folder",
            "dataset_source": args.dataset_slug if args.dataset_root is None else "local-path",
            "dataset_root": str(dataset_root.resolve()),
            "image_size": int(args.image_size),
            "feature_count": int(X.shape[1]),
            "class_labels": [int(x) for x in classes.tolist()],
            "class_distribution": class_distribution,
            "metrics": {
                "decision_tree_accuracy": float(dt_accuracy),
                "random_forest_accuracy": float(rf_accuracy),
                "random_forest_oob_score": float(random_forest.oob_score_),
            },
        },
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, args.model_path, compress=3)
    print(f"\n[6/6] Modelo guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
