"""
App educativa para detectar números escritos (0..9) con 4 modelos:
1) Árbol de Decisión
2) Random Forest
3) MLP pequeña
4) MLP profunda (regularizada)

El usuario puede seleccionar cualquier combinación (1/4, 2/4, 3/4 o 4/4).
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    from streamlit_drawable_canvas import st_canvas

    CANVAS_IMPORT_ERROR = None
except ModuleNotFoundError as import_error:
    st_canvas = None
    CANVAS_IMPORT_ERROR = import_error

MODEL_PATH = Path("modelo.joblib")
BLANK_THRESHOLD = 245
MIN_INK_PIXELS = 35

MODEL_REGISTRY = {
    "Árbol de Decisión": "decision_tree",
    "Random Forest": "random_forest",
    "MLP pequeña (1 capa oculta)": "mlp_small",
    "MLP profunda (2 capas ocultas)": "mlp_deep",
}


@st.cache_resource
def load_artifact(model_path: Path) -> dict:
    """Carga el modelo y valida que existan las claves mínimas esperadas."""
    artifact = joblib.load(model_path)
    expected_keys = {"decision_tree", "random_forest", "metadata"}
    missing_keys = expected_keys.difference(artifact.keys())
    if missing_keys:
        raise KeyError(f"Faltan claves requeridas en el modelo: {sorted(missing_keys)}")
    return artifact


def preprocess_pil_image(input_image: Image.Image, image_size: int) -> tuple[Image.Image, np.ndarray]:
    """
    Preprocesado único para todos los modelos:
    - Escala de grises
    - Reducción de resolución
    - Normalización a [0, 1]
    - Vector 1D
    """
    grayscale = ImageOps.grayscale(input_image)
    resized = grayscale.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    feature_vector = image_array.reshape(1, -1)
    return resized, feature_vector


def canvas_result_to_pil(canvas_image_data: np.ndarray) -> Image.Image:
    """Convierte el array RGBA del canvas a imagen PIL RGB."""
    rgba_array = np.asarray(canvas_image_data, dtype=np.uint8)
    return Image.fromarray(rgba_array, mode="RGBA").convert("RGB")


def has_enough_stroke(grayscale_image: Image.Image) -> bool:
    """Evita inferencias con lienzo casi vacío."""
    grayscale_array = np.asarray(grayscale_image, dtype=np.uint8)
    ink_pixels = int(np.count_nonzero(grayscale_array < BLANK_THRESHOLD))
    return ink_pixels >= MIN_INK_PIXELS


def predict_with_probabilities(model, vector: np.ndarray) -> tuple[int, float, np.ndarray, list[int]]:
    """Devuelve predicción, confianza y distribución de probabilidad por clase."""
    probabilities = model.predict_proba(vector)[0]
    model_classes = [int(value) for value in model.classes_]
    predicted_label = int(model.predict(vector)[0])
    predicted_index = model_classes.index(predicted_label)
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities, model_classes


def get_forest_votes(random_forest, vector: np.ndarray, class_labels: list[int]) -> np.ndarray:
    """Recuento de votos por clase para visualizar el voto mayoritario del bosque."""
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
) -> None:
    """Renderiza resultados de un único modelo seleccionado."""
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


def render_predictions(
    selected_model_names: list[str],
    artifact: dict,
    model_input: np.ndarray,
    class_labels: list[int],
) -> None:
    """
    Ejecuta inferencia de todos los modelos seleccionados y construye tabla comparativa.
    """
    st.subheader("Predicciones de los modelos seleccionados")
    comparison_rows = []

    for display_name in selected_model_names:
        model_key = MODEL_REGISTRY[display_name]
        if model_key not in artifact:
            st.warning(
                f"El modelo '{display_name}' no existe en `modelo.joblib`. "
                "Reentrena con `python train.py` para incluir los 4 modelos."
            )
            continue

        model = artifact[model_key]
        predicted_label, confidence, probabilities, model_classes = predict_with_probabilities(model, model_input)

        comparison_rows.append(
            {
                "Modelo": display_name,
                "Predicción": int(predicted_label),
                "Confianza": f"{confidence:.2%}",
            }
        )

        with st.container(border=True):
            render_model_block(
                model_display_name=display_name,
                model_key=model_key,
                model=model,
                model_input=model_input,
                class_labels=class_labels,
            )

    if comparison_rows:
        st.markdown("### Comparativa rápida")
        st.dataframe(comparison_rows, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="Digit Detector App (4 modelos)",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ Detector de números escritos (0..9)")
st.caption(
    "Comparativa educativa de 4 modelos: Árbol, Random Forest, MLP pequeña y MLP profunda."
)

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

with st.expander("📘 Conceptos de ML usados en la app", expanded=False):
    st.markdown(
        """
        - **Impureza de Gini:** criterio de división de Árbol y Random Forest.
        - **Bootstrapping + Voto mayoritario:** lógica central del Random Forest.
        - **Redes neuronales (MLP):** dos variantes de capacidad distinta para comparar sesgo/varianza.
        - **Mismo preprocesado para todos:** comparación justa entre 4 modelos sobre la misma entrada.
        """
    )

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Acc Árbol",
    f"{metrics.get('decision_tree', {}).get('accuracy', 0.0):.3f}",
)
metric_columns[1].metric(
    "Acc Bosque",
    f"{metrics.get('random_forest', {}).get('accuracy', 0.0):.3f}",
)
metric_columns[2].metric(
    "Acc MLP pequeña",
    f"{metrics.get('mlp_small', {}).get('accuracy', 0.0):.3f}",
)
metric_columns[3].metric(
    "Acc MLP profunda",
    f"{metrics.get('mlp_deep', {}).get('accuracy', 0.0):.3f}",
)

if "oob_score" in metrics.get("random_forest", {}):
    st.caption(f"OOB Random Forest: {metrics['random_forest']['oob_score']:.3f}")

st.sidebar.header("⚙️ Configuración de inferencia")
selected_model_names = st.sidebar.multiselect(
    "Selecciona 1 a 4 modelos para ejecutar:",
    options=list(MODEL_REGISTRY.keys()),
    default=["Árbol de Decisión", "Random Forest"],
    max_selections=4,
)

if not selected_model_names:
    st.warning("Selecciona al menos un modelo para ejecutar inferencia.")
    st.stop()

drawing_mode = st.sidebar.selectbox(
    "Herramienta del lienzo:",
    ("freedraw", "line", "rect", "circle", "polygon", "point", "text"),
)
stroke_width = st.sidebar.slider("Grosor del trazo", 1, 30, 12)
stroke_color = st.sidebar.color_picker("Color del trazo", "#000000")
background_color = st.sidebar.color_picker("Color de fondo", "#FFFFFF")
realtime_update = st.sidebar.checkbox("Actualizar en tiempo real", True)

point_display_radius = 4
font_size = 20
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Radio del punto", 1, 30, 4)
if drawing_mode == "text":
    font_size = st.sidebar.slider("Tamaño de fuente", 8, 72, 20)

tab_canvas, tab_upload = st.tabs(["✍️ Dibujar número", "🖼️ Subir imagen"])

with tab_canvas:
    st.write("Dibuja un número del **0 al 9** y compara los modelos seleccionados.")
    if st_canvas is None:
        st.error(
            "No se pudo importar `streamlit_drawable_canvas`. "
            "Instala dependencias en el entorno correcto con:\n\n"
            "`python -m pip install -r requirements.txt`"
        )
        st.code(sys.executable, language="text")
        st.caption(
            f"Detalle técnico: {CANVAS_IMPORT_ERROR}. "
            "Ejecuta la app con `.venv\\Scripts\\python.exe -m streamlit run app.py`."
        )
    else:
        canvas = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=background_color,
            update_streamlit=realtime_update,
            height=320,
            width=320,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            font_size=font_size if drawing_mode == "text" else 20,
            return_image_data=True,
            key="digit-canvas",
        )

        if canvas.image_data is not None:
            pil_canvas = canvas_result_to_pil(canvas.image_data)
            processed_image, model_input = preprocess_pil_image(pil_canvas, image_size=image_size)
            has_stroke = has_enough_stroke(processed_image)

            col_left, col_right = st.columns([1, 1.3])
            with col_left:
                st.subheader("Vista previa")
                st.image(pil_canvas, caption="Lienzo original", use_container_width=True)
                st.image(
                    processed_image,
                    caption=f"Procesada ({image_size}x{image_size}, gris)",
                    use_container_width=True,
                )

            with col_right:
                if has_stroke:
                    render_predictions(
                        selected_model_names=selected_model_names,
                        artifact=artifact,
                        model_input=model_input,
                        class_labels=class_labels,
                    )
                else:
                    st.info("Dibuja más trazos (se detectó lienzo casi vacío).")

        if canvas.json_data is not None:
            with st.expander("Datos vectoriales del dibujo (json_data)", expanded=False):
                st.json(canvas.json_data)

with tab_upload:
    st.write("Sube una imagen de un número escrito para comparar modelos.")
    uploaded_file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"], key="upload-image")

    if uploaded_file is not None:
        with Image.open(uploaded_file) as uploaded_img:
            original_uploaded = uploaded_img.convert("RGB")
        processed_image, model_input = preprocess_pil_image(original_uploaded, image_size=image_size)

        col_left, col_right = st.columns([1, 1.3])
        with col_left:
            st.subheader("Imagen subida")
            st.image(original_uploaded, caption="Original", use_container_width=True)
            st.image(
                processed_image,
                caption=f"Procesada ({image_size}x{image_size}, gris)",
                use_container_width=True,
            )
        with col_right:
            render_predictions(
                selected_model_names=selected_model_names,
                artifact=artifact,
                model_input=model_input,
                class_labels=class_labels,
            )
