"""
App educativa para detectar números escritos (0..9) con Streamlit.

Entrada principal:
- Lienzo de dibujo con streamlit-drawable-canvas
Entrada opcional:
- Subida de imagen
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


@st.cache_resource
def load_artifact(model_path: Path) -> dict:
    """Carga el modelo y valida la estructura mínima del artefacto."""
    artifact = joblib.load(model_path)
    expected_keys = {"decision_tree", "random_forest", "metadata"}
    missing_keys = expected_keys.difference(artifact.keys())
    if missing_keys:
        raise KeyError(f"Faltan claves requeridas en el modelo: {sorted(missing_keys)}")
    return artifact


def preprocess_pil_image(input_image: Image.Image, image_size: int) -> tuple[Image.Image, np.ndarray]:
    """
    Convierte una imagen PIL al formato esperado por el modelo:
    - Escala de grises
    - Redimensión al tamaño del entrenamiento
    - Normalización [0, 1]
    - Vector fila (1, n_features)
    """
    grayscale = ImageOps.grayscale(input_image)
    resized = grayscale.resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    feature_vector = image_array.reshape(1, -1)
    return resized, feature_vector


def canvas_result_to_pil(canvas_image_data: np.ndarray) -> Image.Image:
    """Convierte el array RGBA del canvas a una imagen PIL RGB."""
    rgba_array = np.asarray(canvas_image_data, dtype=np.uint8)
    return Image.fromarray(rgba_array, mode="RGBA").convert("RGB")


def has_enough_stroke(grayscale_image: Image.Image) -> bool:
    """Comprueba si el lienzo tiene suficientes píxeles no blancos para inferir."""
    grayscale_array = np.asarray(grayscale_image, dtype=np.uint8)
    ink_pixels = int(np.count_nonzero(grayscale_array < BLANK_THRESHOLD))
    return ink_pixels >= MIN_INK_PIXELS


def predict_with_probabilities(model, vector: np.ndarray) -> tuple[int, float, np.ndarray, list[int]]:
    """Predicción de clase y confianza basada en predict_proba."""
    probabilities = model.predict_proba(vector)[0]
    model_classes = [int(value) for value in model.classes_]
    predicted_label = int(model.predict(vector)[0])
    predicted_index = model_classes.index(predicted_label)
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities, model_classes


def get_forest_votes(random_forest, vector: np.ndarray, class_labels: list[int]) -> np.ndarray:
    """Recuento de votos individuales de cada árbol del bosque."""
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    vote_counts = np.zeros(len(class_labels), dtype=np.int32)
    for estimator in random_forest.estimators_:
        vote = int(estimator.predict(vector)[0])
        if vote in label_to_index:
            vote_counts[label_to_index[vote]] += 1
    return vote_counts


def render_prediction_panel(
    model_mode: str,
    decision_tree,
    random_forest,
    class_labels: list[int],
    model_input: np.ndarray,
) -> None:
    """Renderiza resultados, comparativa y gráficos de voto/probabilidades."""
    st.subheader("Predicción")
    comparison_rows = []

    if model_mode in ("Solo Árbol de Decisión", "Comparar ambos"):
        dt_label, dt_confidence, _, _ = predict_with_probabilities(decision_tree, model_input)
        st.success(f"Árbol de Decisión → {dt_label}")
        st.write(f"Confianza estimada: **{dt_confidence:.2%}**")
        comparison_rows.append(
            {
                "Modelo": "Árbol de Decisión",
                "Predicción": int(dt_label),
                "Confianza": f"{dt_confidence:.2%}",
            }
        )

    if model_mode in ("Solo Random Forest", "Comparar ambos"):
        rf_label, rf_confidence, rf_probabilities, rf_model_classes = predict_with_probabilities(
            random_forest, model_input
        )
        st.success(f"Random Forest → {rf_label}")
        st.write(f"Confianza estimada: **{rf_confidence:.2%}**")
        comparison_rows.append(
            {
                "Modelo": "Random Forest",
                "Predicción": int(rf_label),
                "Confianza": f"{rf_confidence:.2%}",
            }
        )

        st.markdown("#### 🗳️ Voto Mayoritario del Random Forest")
        vote_counts = get_forest_votes(random_forest, model_input, class_labels=class_labels)
        votes_chart = {
            "Dígito": [str(label) for label in class_labels],
            "Votos": [int(count) for count in vote_counts],
        }
        st.bar_chart(votes_chart, x="Dígito", y="Votos", use_container_width=True)
        st.caption(f"Votos totales emitidos por los árboles: {int(vote_counts.sum())}")

        st.markdown("#### Probabilidad por clase (Random Forest)")
        prob_chart = {
            "Dígito": [str(label) for label in rf_model_classes],
            "Probabilidad": [float(value) for value in rf_probabilities],
        }
        st.bar_chart(prob_chart, x="Dígito", y="Probabilidad", use_container_width=True)

    if comparison_rows:
        st.markdown("#### Comparativa rápida")
        st.dataframe(comparison_rows, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="Digit Detector App",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ Detector de números escritos (0..9)")
st.caption("Proyecto educativo con Árbol de Decisión y Random Forest usando scikit-learn + Streamlit.")

if not MODEL_PATH.exists():
    st.error(
        "No se encontró `modelo.joblib`. "
        "Ejecuta `python train.py` para entrenar y generar el modelo."
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

decision_tree = artifact["decision_tree"]
random_forest = artifact["random_forest"]
metadata = artifact["metadata"]
image_size = int(metadata["image_size"])
class_labels = [int(label) for label in metadata["class_labels"]]
metrics = metadata.get("metrics", {})

with st.expander("📘 Conceptos ML en uso", expanded=False):
    st.markdown(
        """
        - **Impureza de Gini:** criterio de división en los nodos del árbol.
        - **Bootstrapping:** cada árbol del bosque se entrena con una muestra aleatoria con reemplazo.
        - **Feature Bagging:** en cada split del bosque se evalúa un subconjunto de píxeles.
        - **Voto Mayoritario:** la clase final del bosque es la más votada por sus árboles.
        - **OOB Score:** estimación interna del rendimiento usando muestras fuera de bolsa.
        """
    )

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Accuracy Árbol", f"{metrics.get('decision_tree_accuracy', 0.0):.3f}")
m_col2.metric("Accuracy Bosque", f"{metrics.get('random_forest_accuracy', 0.0):.3f}")
m_col3.metric("OOB Bosque", f"{metrics.get('random_forest_oob_score', 0.0):.3f}")

st.sidebar.header("⚙️ Configuración de inferencia")
model_mode = st.sidebar.selectbox(
    "Modelo a ejecutar:",
    ("Solo Árbol de Decisión", "Solo Random Forest", "Comparar ambos"),
)
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
    st.write("Dibuja un número del **0 al 9** y obtén la predicción.")
    if st_canvas is None:
        st.error(
            "No se pudo importar `streamlit_drawable_canvas`. "
            "Instala dependencias en el mismo entorno con:\n\n"
            "`python -m pip install -r requirements.txt`"
        )
        st.code(sys.executable, language="text")
        st.caption(
            f"Detalle técnico: {CANVAS_IMPORT_ERROR}. "
            "Lanza la app con el Python correcto, por ejemplo: "
            "`.venv\\Scripts\\python.exe -m streamlit run app.py`"
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

            col_left, col_right = st.columns([1, 1.1])
            with col_left:
                st.subheader("Vista previa")
                st.image(pil_canvas, caption="Lienzo original", use_container_width=True)
                st.image(
                    processed_image,
                    caption=f"Procesada para el modelo ({image_size}x{image_size}, gris)",
                    use_container_width=True,
                )

            with col_right:
                if has_stroke:
                    render_prediction_panel(
                        model_mode=model_mode,
                        decision_tree=decision_tree,
                        random_forest=random_forest,
                        class_labels=class_labels,
                        model_input=model_input,
                    )
                else:
                    st.info("Dibuja más trazos (se detectó lienzo casi vacío).")

        if canvas.json_data is not None:
            with st.expander("Datos vectoriales del dibujo (json_data)", expanded=False):
                st.json(canvas.json_data)

with tab_upload:
    st.write("También puedes probar con una imagen guardada de un número escrito.")
    uploaded_file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"], key="upload-image")
    if uploaded_file is not None:
        with Image.open(uploaded_file) as uploaded_img:
            original_uploaded = uploaded_img.convert("RGB")
        processed_image, model_input = preprocess_pil_image(original_uploaded, image_size=image_size)

        col_left, col_right = st.columns([1, 1.1])
        with col_left:
            st.subheader("Imagen subida")
            st.image(original_uploaded, caption="Original", use_container_width=True)
            st.image(
                processed_image,
                caption=f"Procesada para el modelo ({image_size}x{image_size}, gris)",
                use_container_width=True,
            )
        with col_right:
            render_prediction_panel(
                model_mode=model_mode,
                decision_tree=decision_tree,
                random_forest=random_forest,
                class_labels=class_labels,
                model_input=model_input,
            )
