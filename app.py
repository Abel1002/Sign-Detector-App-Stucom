"""
App educativa para clasificar dígitos escritos (0..9) con 5 modelos:
1) Árbol de Decisión
2) Random Forest
3) MLP básica
4) MLP tratada (Solución 1: Max Pooling)
5) MLP tratada (Solución 3: Atención espacial)

Entrada de usuario: solo subida de imagen (sin lienzo).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = Path("modelo.joblib")

MODEL_REGISTRY = {
    "Árbol de Decisión": "decision_tree",
    "Random Forest": "random_forest",
    "MLP básica": "mlp_basic",
    "MLP tratada (Solución 1: Max Pooling)": "mlp_sol1_maxpool",
    "MLP tratada (Solución 3: Atención espacial)": "mlp_sol3_attention",
}


@st.cache_resource
def load_artifact(model_path: Path) -> dict:
    """Carga el artefacto de modelos y valida su estructura base."""
    artifact = joblib.load(model_path)
    expected_keys = {"decision_tree", "random_forest", "metadata"}
    missing_keys = expected_keys.difference(artifact.keys())
    if missing_keys:
        raise KeyError(f"Faltan claves requeridas en el modelo: {sorted(missing_keys)}")
    return artifact


def preprocess_pil_image(input_image: Image.Image, image_size: int) -> tuple[Image.Image, np.ndarray]:
    """
    Preprocesado compartido entre entrenamiento e inferencia:
    - gris
    - resize
    - normalización [0, 1]
    - flatten 1D
    """
    grayscale = ImageOps.grayscale(input_image)
    resized = grayscale.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    feature_vector = image_array.reshape(1, -1)
    return resized, feature_vector


def predict_with_probabilities(model, vector: np.ndarray) -> tuple[int, float, np.ndarray, list[int]]:
    """Predicción de clase + confianza + distribución completa de probabilidad."""
    probabilities = model.predict_proba(vector)[0]
    model_classes = [int(value) for value in model.classes_]
    predicted_label = int(model.predict(vector)[0])
    predicted_index = model_classes.index(predicted_label)
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities, model_classes


def build_global_maxpool_features(vector: np.ndarray, image_size: int) -> np.ndarray:
    """Replica en inferencia la Solución 1 (max pooling espacial 2x2)."""
    if image_size < 2:
        return vector
    even_size = image_size if image_size % 2 == 0 else image_size - 1
    image = vector.reshape(1, image_size, image_size)[:, :even_size, :even_size]
    pooled = image.reshape(1, even_size // 2, 2, even_size // 2, 2).max(axis=(2, 4))
    return pooled.reshape(1, -1)


def build_attention_features(vector: np.ndarray, image_size: int) -> np.ndarray:
    """Replica en inferencia la Solución 3 (máscara de atención espacial)."""
    image = vector.reshape(image_size, image_size)
    min_value = float(image.min())
    max_value = float(image.max())
    dynamic_range = max(max_value - min_value, 1e-6)
    normalized = (image - min_value) / dynamic_range
    attention_mask = 0.25 + 0.75 * normalized
    weighted = image * attention_mask
    return weighted.reshape(1, -1)


def get_forest_votes(random_forest, vector: np.ndarray, class_labels: list[int]) -> np.ndarray:
    """Cuenta cuántos árboles del bosque votan cada clase."""
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    vote_counts = np.zeros(len(class_labels), dtype=np.int32)
    for estimator in random_forest.estimators_:
        vote = int(estimator.predict(vector)[0])
        if vote in label_to_index:
            vote_counts[label_to_index[vote]] += 1
    return vote_counts


def render_model_block(
    model_display_name: str,
    model_key: str,
    model,
    model_input: np.ndarray,
    class_labels: list[int],
) -> tuple[int, float]:
    """Muestra salida visual de un modelo y devuelve predicción/confianza."""
    predicted_label, confidence, probabilities, model_classes = predict_with_probabilities(model, model_input)
    st.markdown(f"### {model_display_name}")
    st.success(f"Predicción: **{predicted_label}**")
    st.write(f"Confianza estimada: **{confidence:.2%}**")

    probability_chart_data = {
        "Dígito": [str(label) for label in model_classes],
        "Probabilidad": [float(value) for value in probabilities],
    }
    st.bar_chart(probability_chart_data, x="Dígito", y="Probabilidad", use_container_width=True)

    if model_key == "random_forest":
        vote_counts = get_forest_votes(model, model_input, class_labels=class_labels)
        vote_chart_data = {
            "Dígito": [str(label) for label in class_labels],
            "Votos": [int(count) for count in vote_counts],
        }
        st.markdown("#### Voto mayoritario del Random Forest")
        st.bar_chart(vote_chart_data, x="Dígito", y="Votos", use_container_width=True)
        st.caption(f"Votos totales emitidos: {int(vote_counts.sum())}")

    return predicted_label, confidence


def render_predictions(
    selected_model_names: list[str],
    artifact: dict,
    model_input_base: np.ndarray,
    image_size: int,
    class_labels: list[int],
) -> None:
    """Ejecuta todos los modelos seleccionados y renderiza comparativa."""
    st.subheader("Predicciones de los modelos seleccionados")
    comparison_rows = []

    for display_name in selected_model_names:
        model_key = MODEL_REGISTRY[display_name]
        if model_key not in artifact:
            st.warning(
                f"El modelo '{display_name}' no existe en `modelo.joblib`. "
                "Reentrena con `python train.py` para incluir los modelos actuales."
            )
            continue

        model = artifact[model_key]
        # Selección de representación de entrada según el modelo neuronal.
        if model_key == "mlp_sol1_maxpool":
            model_input = build_global_maxpool_features(model_input_base, image_size=image_size)
        elif model_key == "mlp_sol3_attention":
            model_input = build_attention_features(model_input_base, image_size=image_size)
        else:
            model_input = model_input_base

        with st.container(border=True):
            predicted_label, confidence = render_model_block(
                model_display_name=display_name,
                model_key=model_key,
                model=model,
                model_input=model_input,
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


st.set_page_config(
    page_title="Digit Detector App (5 modelos)",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 Detector de números escritos (0..9)")
st.caption("Comparativa educativa de 5 modelos con selección libre (1 a 5 modelos).")

if not MODEL_PATH.exists():
    st.error(
        "No se encontró `modelo.joblib`. "
        "Ejecuta `python train.py` para entrenar y generar el artefacto."
    )
    st.stop()

try:
    artifact = load_artifact(MODEL_PATH)
except FileNotFoundError:
    st.error("No se pudo abrir el archivo `modelo.joblib`.")
    st.stop()
except KeyError as error:
    st.error(f"Formato de modelo inválido: {error}")
    st.stop()

metadata = artifact["metadata"]
image_size = int(metadata["image_size"])
class_labels = [int(label) for label in metadata["class_labels"]]
metrics = metadata.get("metrics", {})

with st.expander("📘 Conceptos de ML usados", expanded=False):
    st.markdown(
        """
        - **Árbol de Decisión:** modelo base simple e interpretable.
        - **Random Forest:** ensamble por voto mayoritario con bootstrapping.
        - **MLP básica:** baseline neuronal para comparar contra clásicos.
        - **MLP Solución 1:** usa max pooling espacial para ser menos sensible al desplazamiento.
        - **MLP Solución 3:** usa ponderación espacial dinámica del trazo.
        - **Mismo preprocesado para todos:** comparación justa entre modelos.
        """
    )

metric_columns = st.columns(4)
metric_columns[0].metric("Acc Árbol", f"{metrics.get('decision_tree', {}).get('accuracy', 0.0):.3f}")
metric_columns[1].metric("Acc Bosque", f"{metrics.get('random_forest', {}).get('accuracy', 0.0):.3f}")
metric_columns[2].metric("Acc MLP básica", f"{metrics.get('mlp_basic', {}).get('accuracy', 0.0):.3f}")
metric_columns[3].metric(
    "Acc MLP Solución 3",
    f"{metrics.get('mlp_sol3_attention', {}).get('accuracy', 0.0):.3f}",
)
st.caption(f"Acc MLP Solución 1: {metrics.get('mlp_sol1_maxpool', {}).get('accuracy', 0.0):.3f}")

if "oob_score" in metrics.get("random_forest", {}):
    st.caption(f"OOB Random Forest: {metrics['random_forest']['oob_score']:.3f}")

st.sidebar.header("⚙️ Configuración de inferencia")
selected_model_names = st.sidebar.multiselect(
    "Selecciona 1 a 5 modelos para ejecutar:",
    options=list(MODEL_REGISTRY.keys()),
    default=["Árbol de Decisión", "Random Forest", "MLP básica"],
    max_selections=5,
)

if not selected_model_names:
    st.warning("Selecciona al menos un modelo para ejecutar inferencia.")
    st.stop()

uploaded_file = st.file_uploader("Sube una imagen de un número (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with Image.open(uploaded_file) as uploaded_img:
        original_uploaded = uploaded_img.convert("RGB")
    processed_image, model_input = preprocess_pil_image(original_uploaded, image_size=image_size)

    col_left, col_right = st.columns([1, 1.4])
    with col_left:
        st.subheader("Imagen de entrada")
        st.image(original_uploaded, caption="Original", use_container_width=True)
        st.image(
            processed_image,
            caption=f"Procesada para inferencia ({image_size}x{image_size}, gris)",
            use_container_width=True,
        )

    with col_right:
        render_predictions(
            selected_model_names=selected_model_names,
            artifact=artifact,
            model_input_base=model_input,
            image_size=image_size,
            class_labels=class_labels,
        )
else:
    st.info("Sube una imagen para ejecutar la predicción con los modelos seleccionados.")
