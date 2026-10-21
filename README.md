# Sign Detector App Stucom

Clasificación de números escritos (0..9) con comparación entre:

- **Modelos clásicos**: Árbol de Decisión y Random Forest
- **Redes neuronales (3)**:
  1. MLP básica
  2. CNN tratada (**Solución 1 real**: Global Max Pooling sobre mapas convolucionales)
  3. MLP tratada (**Solución 3**: atención espacial dinámica)

---

## 1) Dataset

Por defecto se descarga:

```python
import kagglehub
path = kagglehub.dataset_download("karnikakapoor/digits")
print(path)
```

También puedes usar ruta local (carpetas `0..9`).

---

## 2) Arquitectura resumida

### Clásicos + MLP tabulares

Preprocesado:
1. Grises
2. Resize 28x28
3. Normalización [0,1]
4. Flatten

Se usa para:
- `decision_tree`
- `random_forest`
- `mlp_basic`
- `mlp_sol3_attention` (después de aplicar máscara espacial)

### Solución 1 real (CNN)

Se entrena una red PyTorch:

- Backbone: `MobileNetV2.features`
- Pooling: `nn.AdaptiveMaxPool2d((1,1))` (**Global Max Pooling real**)
- Clasificador: `Dropout(0.2) + Linear(1280 -> 10)`

Esto implementa la teoría correcta: el pooling global actúa sobre **canales convolucionales**, no sobre píxeles crudos.

---

## 3) Parámetros principales usados

## Random Forest
- `n_estimators=140`
- `max_features="sqrt"`
- `oob_score=True`

## MLP básica
- `hidden_layer_sizes=(128,)`
- `activation="relu"`
- `solver="adam"`
- `alpha=1e-4`
- `learning_rate_init=1e-3`

## MLP Solución 3
- `hidden_layer_sizes=(256, 128)`
- `solver="adam"`
- `alpha=5e-4`
- `learning_rate_init=8e-4`

## CNN Solución 1
- `cnn_image_size=128`
- `cnn_epochs=4` (por defecto)
- `cnn_batch_size=32`
- Optimizador: `Adam` sobre la cabeza clasificadora
- `RandomAffine` en train (rotación/desplazamiento/escala) para robustez

---

## 4) ¿Por qué `solver="adam"` en MLP?

Porque es estable y práctico para este tipo de datos:
- menos ajuste manual que SGD,
- buena convergencia en escenarios ruidosos/no lineales.

Alternativas:
- `sgd`: más control (momentum, schedules), útil para tuning fino.
- `lbfgs`: puede funcionar bien en datasets pequeños, escala peor.

---

## 5) Si el problema fuera texto

### En clásico:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Naive Bayes

### En deep learning:
- LSTM/GRU
- Transformers (BERT-like)
- Optimizador frecuente: **AdamW**

---

## 6) Soluciones Alternativas a "DL"

No aplicadas aquí para priorizar práctica del modelo, pero válidas:

1. **Centrado previo del trazo** (bounding box + recentrado en app).
2. **Aumento de datos con desplazamiento** durante entrenamiento.

Estas estrategias no son peores ni mejores por sí mismas; depende de objetivo, coste y datos.

---

## 7) Entrenamiento

```powershell
cd c:\Users\abel.perez\Desktop\ML\SignDetectorApp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

Opciones:

```powershell
python train.py --max-images-per-class 300
python train.py --cnn-epochs 6
python train.py --dataset-root "C:\ruta\dataset"
```

---

## 8) Ejecución de la app

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

La app permite seleccionar cualquier combinación de 1 a 5 modelos.

---

## 9) Render

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Necesitas subir `modelo.joblib` para inferencia directa en deploy.
