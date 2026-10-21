# Digit Detector App (4 modelos)

Proyecto educativo para clasificar números escritos (0..9) con **4 modelos**:

1. Árbol de Decisión
2. Random Forest
3. Red neuronal MLP pequeña
4. Red neuronal MLP profunda

La app en Streamlit permite seleccionar y comparar **1, 2, 3 o 4 modelos** en la misma inferencia.

---

## 1) Estructura del proyecto

```text
SignDetectorApp/
├── app.py
├── train.py
├── modelo.joblib
├── requirements.txt
├── .gitignore
├── render.yaml
├── README.md
└── dataset/            # opcional (si quieres entrenar desde ruta local)
```

---

## 2) Dataset usado

Por defecto se descarga automáticamente:

```python
import kagglehub
path = kagglehub.dataset_download("karnikakapoor/digits")
print(path)
```

También puedes usar un dataset local con carpetas `0..9`.

---

## 3) Preprocesado de imágenes (igual para los 4 modelos)

Pipeline en [train.py](C:/Users/abel.perez/Desktop/ML/SignDetectorApp/train.py) y [app.py](C:/Users/abel.perez/Desktop/ML/SignDetectorApp/app.py):

1. Convertir a escala de grises.
2. Redimensionar a `image_size x image_size` (por defecto `28x28`).
3. Normalizar píxeles a rango `[0, 1]`.
4. Aplanar (flatten) a vector 1D.

Con `28x28`, cada imagen pasa a 784 features.

---

## 4) Modelos y parámetros usados

## 4.1 Árbol de Decisión

- `criterion="gini"`: minimiza impureza de clases en cada split.
- `max_depth=18`: limita profundidad para reducir sobreajuste.
- `min_samples_leaf=2`: evita hojas demasiado pequeñas.
- `random_state=42`: reproducibilidad.

## 4.2 Random Forest

- `n_estimators=140`: cantidad de árboles.
- `criterion="gini"`: mismo criterio de split.
- `max_depth=22`: profundidad por árbol.
- `max_features="sqrt"`: feature bagging por split.
- `bootstrap=True`: muestreo con reemplazo.
- `oob_score=True`: estimación interna de generalización.
- `n_jobs=-1`: usa todos los cores.
- `random_state=42`: reproducibilidad.

## 4.3 Red neuronal MLP pequeña

- `hidden_layer_sizes=(128,)`: 1 capa oculta de 128 neuronas.
- `activation="relu"`: activación estándar en MLP modernas.
- `solver="adam"`: optimizador adaptativo.
- `alpha=1e-4`: regularización L2.
- `learning_rate_init=1e-3`: tasa inicial.
- `max_iter=60`: iteraciones máximas.
- `early_stopping=True`: para evitar sobreentrenamiento.
- `n_iter_no_change=8`: paciencia de parada.

## 4.4 Red neuronal MLP profunda

- `hidden_layer_sizes=(256, 128)`: 2 capas ocultas.
- `activation="relu"`
- `solver="adam"`
- `alpha=5e-4`: regularización más fuerte.
- `learning_rate_init=8e-4`: paso algo más conservador.
- `max_iter=80`
- `early_stopping=True`
- `n_iter_no_change=10`

---

## 5) ¿Por qué usamos `solver="adam"`?

En este proyecto, Adam se eligió porque:

- funciona bien con datos ruidosos y no lineales;
- suele converger más rápido y de forma más estable que SGD sin tuning complejo;
- es una opción segura para alumnos en primeras prácticas.

### Alternativas válidas para problemas parecidos

- `solver="sgd"`:
  - más control fino (momentum, schedules),
  - útil en experimentación avanzada de aprendizaje.
- `solver="lbfgs"`:
  - puede ir muy bien en datasets pequeños,
  - pero escala peor cuando crecen datos/parámetros.

---

## 6) Si en vez de imágenes clasificamos texto

### En enfoque clásico (rápido y robusto):

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Naive Bayes

### En deep learning:

- Embeddings + LSTM/GRU
- Transformers (BERT-like) + fine-tuning
- Optimizador habitual: AdamW (especialmente en Transformers)

---

## 7) Entrenamiento

```powershell
cd c:\Users\abel.perez\Desktop\ML\SignDetectorApp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

Opcional:

```powershell
python train.py --dataset-root "C:\ruta\dataset"
python train.py --max-images-per-class 500
python train.py --image-size 28
```

Esto genera/actualiza `modelo.joblib`.

---

## 8) Ejecución local de la app

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Luego abre `http://localhost:8501`.

Funcionalidades:
- Subida de imagen.
- Selección libre de modelos (1 a 4).
- Comparativa de predicción y confianza.
- Visualización de probabilidades por clase.
- Visualización de voto mayoritario (Random Forest).

---

## 9) Deploy en Render

## Build Command
```bash
pip install -r requirements.txt
```

## Start Command
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Importante:
- `modelo.joblib` debe estar en el repositorio (o debes entrenarlo en build si prefieres ese flujo).

---

## 10) Notas de rendimiento

- Reducir a 28x28 mantiene el proyecto ligero en RAM/CPU.
- Este enfoque trata píxeles como features tabulares (no convolucional).
- Para visión más compleja, una CNN suele rendir mejor, pero este diseño prioriza simplicidad pedagógica.
