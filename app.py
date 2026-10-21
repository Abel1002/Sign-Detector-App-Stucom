"""
App de inferencia y comparativa de modelos para dígitos 0..9.

Modelos disponibles:
- Árbol de Decisión
- Random Forest
- MLP básica
- CNN Solución 1 (Global Max Pooling real sobre mapas convolucionales)
- MLP Solución 3 (Atención espacial dinámica)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

MODEL_PATH = Path("modelo.joblib")

MODEL_REGISTRY = {
    "Árbol de Decisión": "decision_tree",
    "Random Forest": "random_forest",
    "MLP básica": "mlp_basic",
    "CNN tratada (Solución 1: Global Max Pooling)": "cnn_sol1_global_maxpool",
    "MLP tratada (Solución 3: Atención espacial)": "mlp_sol3_attention",
}


class MobileNetV2GlobalMaxPool(nn.Module):
    """Definición de arquitectura CNN usada por la Solución 1."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.features = base.features
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_max_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@st.cache_resource
def load_artifact(model_path: Path) -> dict:
    """Carga artefacto de entrenamiento."""
    artifact = joblib.load(model_path)
    expected_keys = {"decision_tree", "random_forest", "metadata"}
    missing_keys = expected_keys.difference(artifact.keys())
    if missing_keys:
        raise KeyError(f"Faltan claves requeridas en el modelo: {sorted(missing_keys)}")
    return artifact


@st.cache_resource
def load_cnn_model_from_artifact(_artifact: dict):
    """Reconstruye CNN Solución 1 desde state_dict guardado."""
    if "cnn_sol1_global_maxpool_state_dict" not in _artifact:
        return None
    config = _artifact.get("cnn_sol1_global_maxpool_config", {})
    model = MobileNetV2GlobalMaxPool(num_classes=int(config.get("num_classes", 10)))
    model.load_state_dict(_artifact["cnn_sol1_global_maxpool_state_dict"], strict=True)
    model.eval()
    return model


def preprocess_pil_image(input_image: Image.Image, image_size: int) -> tuple[Image.Image, np.ndarray]:
    """Preprocesado tabular para Árbol/RF/MLPs (gris + resize + flatten)."""
    grayscale = ImageOps.grayscale(input_image)
    resized = grayscale.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return resized, image_array.reshape(1, -1)


def build_attention_features(vector: np.ndarray, image_size: int) -> np.ndarray:
    """Transformación Solución 3: recentrado + atención espacial sobre tinta."""
    image = vector.reshape(image_size, image_size)
    ink = 1.0 - image
    rows, cols = np.where(ink > 0.1)

    if len(rows) > 0:
        y_min, y_max = int(np.min(rows)), int(np.max(rows))
        x_min, x_max = int(np.min(cols)), int(np.max(cols))
        cropped = Image.fromarray((image * 255.0).astype(np.uint8)).crop((x_min, y_min, x_max + 1, y_max + 1))
        width, height = cropped.size
        scale = (image_size * 0.7) / max(width, height, 1)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cropped.resize(new_size, Image.Resampling.BILINEAR)

        centered = Image.new("L", (image_size, image_size), color=255)
        offset_x = (image_size - new_size[0]) // 2
        offset_y = (image_size - new_size[1]) // 2
        centered.paste(resized, (offset_x, offset_y))
        image = np.asarray(centered, dtype=np.float32) / 255.0
        ink = 1.0 - image

    attention_mask = 0.25 + 0.75 * ink
    weighted_ink = np.clip(ink * attention_mask, 0.0, 1.0)
    weighted = 1.0 - weighted_ink
    return weighted.reshape(1, -1)


def build_cnn_input(pil_image: Image.Image, cnn_image_size: int) -> torch.Tensor:
    """Preprocesado CNN (128x128 + 3 canales + normalización ImageNet)."""
    transform_eval = transforms.Compose(
        [
            transforms.Resize((cnn_image_size, cnn_image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform_eval(pil_image.convert("RGB")).unsqueeze(0)
    return tensor


def predict_with_probabilities(model, vector: np.ndarray) -> tuple[int, float, np.ndarray, list[int]]:
    """Predicción para modelos scikit-learn (clase + confianza + distribución)."""
    probabilities = model.predict_proba(vector)[0]
    model_classes = [int(value) for value in model.classes_]
    predicted_label = int(model.predict(vector)[0])
    predicted_index = model_classes.index(predicted_label)
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities, model_classes


def predict_cnn_with_probabilities(model: nn.Module, tensor_input: torch.Tensor, class_labels: list[int]):
    """Predicción para CNN en PyTorch usando softmax."""
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_labels[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities, class_labels


def get_forest_votes(random_forest, vector: np.ndarray, class_labels: list[int]) -> np.ndarray:
    """Cuenta votos por clase en Random Forest."""
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    vote_counts = np.zeros(len(class_labels), dtype=np.int32)
    for estimator in random_forest.estimators_:
        vote = int(estimator.predict(vector)[0])
        if vote in label_to_index:
            vote_counts[label_to_index[vote]] += 1
    return vote_counts


def render_model_result(
    display_name: str,
    model_key: str,
    predicted_label: int,
    confidence: float,
    probabilities: np.ndarray,
    model_classes: list[int],
    artifact: dict,
    model_input_base: np.ndarray,
    class_labels: list[int],
) -> None:
    """Muestra salida visual de un modelo."""
    st.markdown(f"### {display_name}")
    st.success(f"Predicción: **{predicted_label}**")
    st.write(f"Confianza estimada: **{confidence:.2%}**")

    chart_data = {
        "Dígito": [str(label) for label in model_classes],
        "Probabilidad": [float(value) for value in probabilities],
    }
    st.bar_chart(chart_data, x="Dígito", y="Probabilidad", use_container_width=True)

    if model_key == "random_forest":
        vote_counts = get_forest_votes(artifact["random_forest"], model_input_base, class_labels=class_labels)
        vote_chart_data = {
            "Dígito": [str(label) for label in class_labels],
            "Votos": [int(count) for count in vote_counts],
        }
        st.markdown("#### Voto mayoritario del Random Forest")
        st.bar_chart(vote_chart_data, x="Dígito", y="Votos", use_container_width=True)
        st.caption(f"Votos totales emitidos: {int(vote_counts.sum())}")


def render_predictions(
    selected_model_names: list[str],
    artifact: dict,
    cnn_model: nn.Module | None,
    model_input_base: np.ndarray,
    original_pil_image: Image.Image,
    image_size: int,
    cnn_image_size: int,
    class_labels: list[int],
) -> None:
    """Ejecuta y compara todos los modelos seleccionados."""
    st.subheader("Predicciones de los modelos seleccionados")
    comparison_rows = []

    for display_name in selected_model_names:
        model_key = MODEL_REGISTRY[display_name]

        if model_key == "cnn_sol1_global_maxpool":
            if cnn_model is None:
                st.warning("No se encontró la CNN Solución 1 en el artefacto. Reentrena con `python train.py`.")
                continue
            cnn_input = build_cnn_input(original_pil_image, cnn_image_size=cnn_image_size)
            predicted_label, confidence, probabilities, model_classes = predict_cnn_with_probabilities(
                cnn_model,
                cnn_input,
                class_labels=class_labels,
            )
        else:
            if model_key not in artifact:
                st.warning(f"Modelo ausente: {display_name}. Reentrena con `python train.py`.")
                continue
            model = artifact[model_key]
            if model_key == "mlp_sol3_attention":
                model_input = build_attention_features(model_input_base, image_size=image_size)
            else:
                model_input = model_input_base
            predicted_label, confidence, probabilities, model_classes = predict_with_probabilities(model, model_input)

        with st.container(border=True):
            render_model_result(
                display_name=display_name,
                model_key=model_key,
                predicted_label=predicted_label,
                confidence=confidence,
                probabilities=probabilities,
                model_classes=model_classes,
                artifact=artifact,
                model_input_base=model_input_base,
                class_labels=class_labels,
            )

        comparison_rows.append(
            {
                "Modelo": display_name,
                "Predicción": int(predicted_label),
                "Confianza": f"{confidence:.2%}",
            }
        )

    if comparison_rows:
        st.markdown("### Comparativa rápida")
        st.dataframe(comparison_rows, use_container_width=True, hide_index=True)


st.set_page_config(page_title="Digit Detector App", page_icon="🔢", layout="wide")

st.title("🔢 Detector de números escritos (0..9)")
st.caption("Comparativa educativa de modelos clásicos + 3 redes neuronales.")

if not MODEL_PATH.exists():
    st.error("No se encontró `modelo.joblib`. Ejecuta `python train.py`.")
    st.stop()

try:
    artifact = load_artifact(MODEL_PATH)
except FileNotFoundError:
    st.error("No se pudo abrir `modelo.joblib`.")
    st.stop()
except KeyError as error:
    st.error(f"Formato de modelo inválido: {error}")
    st.stop()

cnn_model = load_cnn_model_from_artifact(artifact)

metadata = artifact["metadata"]
image_size = int(metadata["image_size"])
cnn_config = artifact.get("cnn_sol1_global_maxpool_config", {})
cnn_image_size = int(cnn_config.get("cnn_image_size", metadata.get("cnn_image_size", 128)))
class_labels = [int(label) for label in metadata["class_labels"]]
metrics = metadata.get("metrics", {})

with st.expander("📘 Resumen de soluciones", expanded=False):
    st.markdown(
        """
        - **Solución 1 (real):** CNN con **Global Max Pooling** en mapas convolucionales.
        - **Solución 3:** atención espacial dinámica sobre entrada tabular de la MLP.
        - **Mismos datos y etiquetas:** comparación justa entre enfoques.
        """
    )

metric_columns = st.columns(5)
metric_columns[0].metric("Acc Árbol", f"{metrics.get('decision_tree', {}).get('accuracy', 0.0):.3f}")
metric_columns[1].metric("Acc Bosque", f"{metrics.get('random_forest', {}).get('accuracy', 0.0):.3f}")
metric_columns[2].metric("Acc MLP básica", f"{metrics.get('mlp_basic', {}).get('accuracy', 0.0):.3f}")
metric_columns[3].metric(
    "Acc CNN Solución 1",
    f"{metrics.get('cnn_sol1_global_maxpool', {}).get('accuracy', 0.0):.3f}",
)
metric_columns[4].metric(
    "Acc MLP Solución 3",
    f"{metrics.get('mlp_sol3_attention', {}).get('accuracy', 0.0):.3f}",
)

if "oob_score" in metrics.get("random_forest", {}):
    st.caption(f"OOB Random Forest: {metrics['random_forest']['oob_score']:.3f}")

selected_model_names = st.sidebar.multiselect(
    "Selecciona 1 a 5 modelos para ejecutar:",
    options=list(MODEL_REGISTRY.keys()),
    default=["Árbol de Decisión", "Random Forest", "CNN tratada (Solución 1: Global Max Pooling)"],
    max_selections=5,
)

if not selected_model_names:
    st.warning("Selecciona al menos un modelo.")
    st.stop()

uploaded_file = st.file_uploader("Sube una imagen de un número (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with Image.open(uploaded_file) as uploaded_img:
        original_uploaded = uploaded_img.convert("RGB")
    processed_image, model_input_base = preprocess_pil_image(original_uploaded, image_size=image_size)

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        st.subheader("Imagen de entrada")
        st.image(original_uploaded, caption="Original", use_container_width=True)
        st.image(
            processed_image,
            caption=f"Preprocesada tabular ({image_size}x{image_size}, gris)",
            use_container_width=True,
        )

    with col_right:
        render_predictions(
            selected_model_names=selected_model_names,
            artifact=artifact,
            cnn_model=cnn_model,
            model_input_base=model_input_base,
            original_pil_image=original_uploaded,
            image_size=image_size,
            cnn_image_size=cnn_image_size,
            class_labels=class_labels,
        )
else:
    st.info("Sube una imagen para ejecutar la inferencia.")
