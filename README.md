# Sign Detector App Stucom

Proyecto educativo para clasificar números escritos (0..9) con comparación de modelos clásicos y redes neuronales.

## Modelos disponibles

### Modelos clásicos
1. Árbol de Decisión
2. Random Forest

### Redes neuronales (3 en total)
3. **MLP básica** (baseline)
4. **MLP tratada - Solución 1**: reducción espacial por **Global Max Pooling** previo
5. **MLP tratada - Solución 3**: **Atención espacial dinámica** previa

En la app puedes ejecutar cualquier combinación de **1 a 5 modelos**.

---

## 1) Estructura

```text
SignDetectorApp/
├── app.py
├── train.py
├── modelo.joblib
├── requirements.txt
├── .gitignore
├── render.yaml
└── README.md
```

---

## 2) Dataset

Por defecto se usa:

```python
import kagglehub
path = kagglehub.dataset_download("karnikakapoor/digits")
print(path)
```

También puedes entrenar con ruta local por carpetas (`0/`, `1/`, ... `9/`).

---

## 3) Preprocesado base

Todos los modelos parten del mismo preprocesado:

1. Escala de grises.
2. Reducción de resolución (`image_size`, por defecto 28x28).
3. Normalización `[0, 1]`.
4. Flatten a vector tabular.

Así comparamos modelos de forma justa.

---

## 4) Redes neuronales creadas y por qué

## 4.1 MLP básica

Configuración principal:
- `hidden_layer_sizes=(128,)`
- `activation="relu"`
- `solver="adam"`
- `alpha=1e-4`
- `learning_rate_init=1e-3`
- `max_iter=60`
- `early_stopping=True`

Qué hace:
- Es la red de referencia para ver cómo se comporta una MLP simple cuando el número se desplaza.

## 4.2 MLP tratada - Solución 1 (Global Max Pooling previo)

Qué hace:
- Antes de la red, se aplica un pooling espacial 2x2 por máximo.
- Esto favorece capturar picos de activación del trazo aunque el dígito no esté centrado.

Resultado práctico:
- Menor sensibilidad a que el número esté desplazado en el lienzo/imagen.

## 4.3 MLP tratada - Solución 3 (Atención espacial dinámica previa)

Qué hace:
- Se calcula una máscara espacial por imagen que da más peso al trazo y menos al fondo.
- Luego la red recibe la imagen ponderada.

Resultado práctico:
- La red aprende a focalizarse en zonas informativas en vez de depender tanto de posición absoluta.

---

## 5) ¿Por qué usamos `solver="adam"`?

Usamos Adam porque:
- converge de forma estable sin mucha calibración manual;
- se comporta bien en problemas no lineales con ruido;
- es una opción práctica en contexto educativo.

### Alternativas que podrían funcionar

- `solver="sgd"`:
  - más control (momentum, schedules), útil para cursos avanzados.
- `solver="lbfgs"`:
  - bueno en datasets pequeños, pero menos escalable.

---

## 6) ¿Y si el problema fuera texto en vez de imagen?

### En enfoque clásico
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Naive Bayes

### En deep learning
- Embeddings + LSTM/GRU
- Transformers (BERT-like)
- Optimizador típico: AdamW

---

## 7) Soluciones Alternativas a "DL"

Estas opciones **no se aplicaron en esta implementación** porque el objetivo era practicar más sobre modelo y representación, pero eso **no significa que sean peores ni mejores**: depende del contexto.

### Opción 1: Centrar el número antes de predecir (parche en la entrada)

Idea:
1. Detectar dónde está el trazo.
2. Recortar caja mínima del número.
3. Reescalar y pegar centrado en un lienzo nuevo.

Ventaja:
- Solución rápida sin reentrenar modelos.

Uso típico:
- Cuando quieres corregir desplazamiento inmediatamente desde `app.py`.

### Opción 2: Entrenar con datos descentrados (aumento de datos)

Idea:
- Durante entrenamiento, desplazar/rotar/escalar números de forma aleatoria.
- La red aprende que la posición no define la clase.

Ventaja:
- Mayor robustez estructural del modelo.

Uso típico:
- Cuando puedes permitirte reentrenar y buscas robustez real en producción.

---

## 8) Entrenamiento local

```powershell
cd c:\Users\abel.perez\Desktop\ML\SignDetectorApp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

Opciones:

```powershell
python train.py --dataset-root "C:\ruta\dataset"
python train.py --max-images-per-class 500
python train.py --image-size 28
```

---

## 9) Ejecución local (Streamlit)

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

La app:
- recibe imagen subida;
- permite seleccionar 1..5 modelos;
- muestra predicción, confianza y probabilidades;
- muestra además voto mayoritario para Random Forest.

---

## 10) Deploy en Render

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Nota:
- `modelo.joblib` debe estar en el repositorio para inferencia directa.
