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
- Preprocesado de entrada: **recentrado por bounding-box** + máscara de atención espacial sobre tinta

## CNN Solución 1
- `cnn_image_size=128`
- `cnn_epochs>=8` efectivos (si pasas menos, se eleva a 8 internamente)
- `cnn_batch_size=32`
- Fine-tuning progresivo: se descongelan bloques `features[14:]` de MobileNetV2
- Optimizador: `AdamW` con LR diferencial (backbone bajo, cabeza alta)
- Loss: `FocalLoss(gamma=2.0, label_smoothing=0.05)`
- `RandomAffine` en train: `degrees=12`, `translate=(0.35, 0.35)`, `scale=(0.45, 0.95)`, `fill=255`

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

---

## Redes neuronales de Texto/Caracteristicas.

En el ejercicio de imágenes de [train.py](C:/Users/abel.perez/Desktop/ML/SignDetectorApp/train.py), el problema crítico del modelo base era el **sesgo posicional espacial**: la red memoriza coordenadas frecuentes y falla si el número aparece desplazado.  
Por eso se comparó:

- una red base (MLP),
- frente a redes adaptadas (CNN con Global Max Pooling real, y atención espacial dinámica), con optimización moderna y aumento de datos.

En un ejercicio de texto como `alumnos.py` (género de película usando `plot` + `year` + `runtime`), aparece un problema equivalente:

- TF-IDF trata palabras como columnas estáticas (similar a píxeles planos),
- pierde estructura semántica/orden,
- y puede colapsar ante clases minoritarias o sinopsis ambiguas.

A continuación tienes 3 casuísticas análogas para trabajar en clase manteniendo el enfoque:

### Casuística 1: Desbalance severo de géneros (Focal Loss vs Cross-Entropy)

**Relación con teoría:** función de pérdida y optimización (diapositivas de loss/optimizers).  
**Fallo base:** la red aprende a predecir géneros mayoritarios (ej. Drama) porque minimiza error medio global.

- **Red 1 (base):** `MLPClassifier` con Cross-Entropy estándar (vía `solver="adam"`).
- **Red 2 (adaptada):** PyTorch con **Focal Loss** (`gamma=2.0`) + **AdamW**.

**Idea didáctica:** Focal Loss reduce peso de ejemplos fáciles y obliga a aprender más sobre clases minoritarias.

---

### Casuística 2: Sinopsis telegráficas y palabras clave (Global Max Pooling semántico)

**Relación con teoría:** equivalente conceptual a tu Solución 1 en imágenes (Global Max Pooling real).  
**Fallo base:** con TF-IDF muy disperso, textos cortos activan pocas columnas y un MLP denso puede “apagarse”.

- **Red 1 (base):** MLP sobre vector TF-IDF.
- **Red 2 (adaptada):** Transfer Learning con encoder de lenguaje (ej. MiniLM/DistilBERT) y **Global Max Pooling** sobre embedding/token-features.

**Idea didáctica:** el modelo pregunta si aparece una señal semántica fuerte en cualquier parte del texto, sin depender tanto de posición exacta.

---

### Casuística 3: Sinopsis ambiguas o híbridas (Label Smoothing + Dropout)

**Relación con teoría:** regularización/calibración ante etiquetas no puras.  
**Fallo base:** si una película multi-género se fuerza a una sola etiqueta dura, el entrenamiento recibe señales contradictorias.

- **Red 1 (base):** etiquetas duras one-hot + red densa.
- **Red 2 (adaptada):** `Dropout` + `CrossEntropy` con **Label Smoothing**.

**Idea didáctica:** evita sobreconfianza extrema y mejora robustez en casos híbridos.

---

### Ejemplo de código para alumnos (adaptando `alumnos.py`)

```python
"""
Comparativa educativa:
Red base (MLP sklearn) vs Red adaptada (PyTorch con Focal Loss + AdamW)
"""

import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier

# RED 1: BASE
modelo_red_base = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=50,
    early_stopping=True,
    random_state=42
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        log_pt = -self.ce(logits, targets)
        pt = torch.exp(log_pt)
        return (-((1 - pt) ** self.gamma) * log_pt).mean()


class RedClasificadoraTexto(nn.Module):
    def __init__(self, input_dim=3502, num_classes=10):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.red(x)


modelo_red_adaptada = RedClasificadoraTexto(input_dim=3502, num_classes=10)
criterio_focal = FocalLoss(gamma=2.0)
optimizador_adamw = torch.optim.AdamW(
    modelo_red_adaptada.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

---

### Tabla de equivalencias didácticas (Imágenes ↔ Texto)

| Concepto (Imágenes) | Equivalente (Texto) |
|---|---|
| Píxeles 28x28 | Tokens/palabras de sinopsis |
| Flatten 784 | TF-IDF ~3500 columnas |
| Número desplazado en esquina | Sinopsis corta / clase minoritaria |
| MLP denso base | MLP denso sobre TF-IDF |
| CNN + Global Max Pooling / Focal Loss | Embeddings + pooling / Focal Loss + AdamW |
| Accuracy en desplazados | Macro F1 en clases minoritarias |

Esta sección existe para guiar la transferencia conceptual entre visión y NLP en la misma práctica pedagógica.
