# Sign Detector App Stucom

Clasificación de números escritos (0..9) con comparación entre:

- **Modelos clásicos**: Árbol de Decisión y Random Forest
- **Redes neuronales (3)**:
  1. MLP básica
  2. CNN tratada (**Global Max Pooling sobre mapas convolucionales**)
  3. MLP tratada (**atención espacial dinámica + recentrado**)

---

## Índice

- [1) Dataset](#1-dataset)
- [2) Arquitectura resumida](#2-arquitectura-resumida)
- [3) Parámetros principales usados](#3-parámetros-principales-usados)
- [4) ¿Por qué `solver=\"adam\"` en MLP?](#4-por-qué-solveradam-en-mlp)
- [6) Entrenamiento](#6-entrenamiento)
- [7) Ejecución de la app](#7-ejecución-de-la-app)
- [8) Render](#8-render)
- [9) Resultados finales en `Prueba Desplazados`](#9-resultados-finales-en-prueba-desplazados)
- [5) Si el problema fuera texto](#5-si-el-problema-fuera-texto)
- [10) Posibles mejoras de CNN (ejercicio para alumnos que no vinieron el último día).](#10-posibles-mejoras-de-cnn-ejercicio-para-alumnos-que-no-vinieron-el-último-día)
- [11) Redes neuronales de Texto/Caracteristicas.](#11-redes-neuronales-de-textocaracteristicas)
- [12) Soluciones Alternativas a "DL"](#12-soluciones-alternativas-a-dl)

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
- `mlp_attention_recentered` (después de aplicar recentrado + máscara espacial)

### CNN con Global Max Pooling

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

## MLP de Atención Espacial + Recentrado
- `hidden_layer_sizes=(256, 128)`
- `solver="adam"`
- `alpha=5e-4`
- `learning_rate_init=8e-4`
- Preprocesado de entrada: **recentrado por bounding-box** + máscara de atención espacial sobre tinta

## CNN con Global Max Pooling
- `cnn_image_size=64` (default optimizado CPU)
- `cnn_epochs=4` (default)
- `cnn_batch_size=64`
- Fine-tuning quirúrgico: se descongela solo `features[18]` de MobileNetV2
- Optimizador: `AdamW` con LR diferencial (backbone bajo, cabeza alta)
- Loss: `FocalLoss(gamma=2.0, label_smoothing=0.05)`
- `RandomAffine` en train: `degrees=12`, `translate=(0.30, 0.30)`, `scale=(0.50, 0.95)`, `fill=255`
- Carga CNN en memoria: precarga en RAM para evitar I/O de disco en cada época

---

## 4) ¿Por qué `solver="adam"` en MLP?

Porque es estable y práctico para este tipo de datos:
- menos ajuste manual que SGD,
- buena convergencia en escenarios ruidosos/no lineales.

Alternativas:
- `sgd`: más control (momentum, schedules), útil para tuning fino.
- `lbfgs`: puede funcionar bien en datasets pequeños, escala peor.

---

## 6) Entrenamiento

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

## 7) Ejecución de la app

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

La app permite seleccionar cualquier combinación de 1 a 5 modelos.

---

## 8) Render

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

## 9) Resultados finales en `Prueba Desplazados`

Con los modelos finales entrenados y validados sobre la carpeta [Prueba Desplazados](C:/Users/abel.perez/Desktop/ML/SignDetectorApp/Prueba%20Desplazados):

- **MLP tratada (atención + recentrado): `9/10`**
- **CNN tratada (Global Max Pooling): `4/10`**

Lectura rápida del resultado:
- La MLP mejora mucho porque primero normaliza la posición del trazo (recentrado) y después clasifica.
- La CNN depende más de cómo quedó adaptado el backbone preentrenado a este caso concreto (dígitos simples desplazados), y en esta configuración rápida de CPU todavía se queda corta.

---

## 5) Si el problema fuera texto

La idea es mantener la misma estructura didáctica que usamos en imágenes:

- **Red base**: funciona bien en casos típicos, pero falla en escenarios difíciles.
- **Red adaptada**: aplica técnicas de la presentación para corregir ese fallo.

En texto, un caso típico es clasificar género de películas usando:
- sinopsis (texto libre),
- y algunas variables numéricas (por ejemplo año o duración).

### Casuística 1: clases desbalanceadas (Focal Loss vs Cross-Entropy)

**Problema base:** si hay muchas muestras de géneros mayoritarios, la red se acostumbra a predecir siempre esas clases.

- **Base recomendada:** MLP sobre features de texto con pérdida estándar.
- **Adaptación recomendada:** mantener la arquitectura y cambiar entrenamiento a:
  - `Focal Loss` (por ejemplo `gamma=2.0`),
  - optimizador `AdamW`,
  - `weight_decay` desacoplado (por ejemplo `1e-2`).

**Qué se espera observar:** mejora del **Macro F1** en clases minoritarias.

### Casuística 2: sinopsis muy cortas o con vocabulario distinto

**Problema base:** con TF-IDF, si el texto es corto o usa palabras no vistas, gran parte del vector queda en ceros y el modelo pierde señal.

- **Base recomendada:** MLP sobre TF-IDF (rápido y simple).
- **Adaptación recomendada:** embeddings de lenguaje preentrenado + pooling global semántico.
  - enfoque útil: probar `Global Max Pooling` para detectar si aparece una señal fuerte en cualquier parte del texto.

**Qué se espera observar:** mayor robustez cuando cambian palabras o estilo de redacción.

### Casuística 3: etiquetas ambiguas (películas híbridas)

**Problema base:** etiquetas demasiado rígidas generan sobreconfianza y errores cuando la sinopsis mezcla varios géneros.

- **Base recomendada:** clasificación con etiquetas duras.
- **Adaptación recomendada:**
  - `Dropout` (por ejemplo `p=0.3` a `0.4`),
  - `Label Smoothing` (por ejemplo `0.1` a `0.15`).

**Qué se espera observar:** predicciones menos extremas y mejor generalización en casos ambiguos.

### Resumen práctico para clase

Si quieres una ruta clara de experimentación:
1. empezar por MLP + TF-IDF (baseline),
2. medir `accuracy` y `macro_f1`,
3. introducir una mejora cada vez (Focal Loss, AdamW, Dropout, Label Smoothing, embeddings),
4. comparar contra baseline con la misma partición de datos.

---

## 10) Posibles mejoras de CNN (ejercicio para alumnos que no vinieron el último día).

Esta parte se deja como **ejercicio guiado** (sin código en el README) para practicar los conceptos de clase:

1. **Combinar dos tipos de pooling (Avg + Max)**
   - En vez de usar solo Max Pooling, combinar Average Pooling y Max Pooling para conservar más información del trazo.

2. **Entrenamiento en dos fases (fine-tuning progresivo)**
   - Fase 1: entrenar solo la cabeza final.
   - Fase 2: descongelar los últimos bloques convolucionales y ajustar con una tasa de aprendizaje más baja.

3. **Revisar fondo y normalización de entrada**
   - Verificar que el fondo “vacío” del lienzo no esté generando activaciones artificiales.
   - Probar inversión de fondo/trazo cuando tenga sentido para estabilizar la entrada.

4. **Ajustar augmentation para no deformar demasiado**
   - Reducir transformaciones extremas para no destruir trazos pequeños.
   - Mantener desplazamientos realistas, pero con escalas que conserven forma del número.

5. **Equilibrar velocidad vs robustez**
   - Subir ligeramente épocas o muestras por clase cuando el tiempo de clase lo permita para ganar robustez en desplazados.

Objetivo didáctico: que el alumno compare qué cambios mejoran la robustez sin perder el enfoque de optimización en CPU.

---

## 11) Redes neuronales de Texto/Caracteristicas.

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

**Relación con teoría:** equivalente conceptual a tu CNN con Global Max Pooling en imágenes.  
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

### Ejemplo de código para alumnos, adaptado a ejemplo de codigo de alumnos.

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

---

## 12) Soluciones Alternativas a "DL"

Estas soluciones no se centran en “hacer una red más compleja”, pero son totalmente válidas y muy útiles en proyectos reales.
En este proyecto no fueron la vía principal porque queríamos practicar más entrenamiento de redes.

### Opción 1: centrar el número antes de predecir (parche de entrada)

Lógica simple:
1. detectar dónde está el trazo,
2. recortar el área útil,
3. reescalar y pegar en el centro del lienzo.

Ventaja:
- mejora inmediata sin reentrenar todo el modelo.

Cuándo usarla:
- cuando necesitas una mejora rápida en inferencia con bajo coste.

### Opción 2: entrenar con datos descentrados (augmentation)

Lógica simple:
- durante entrenamiento, mover/rotar/escalar el número de forma aleatoria para que la red aprenda que la posición no define la clase.

Parámetros orientativos:
- `translate` en torno a `0.20`,
- `degrees` moderado (por ejemplo `±10`),
- `scale` controlado para no destruir el trazo.

Ventaja:
- el modelo se vuelve robusto por aprendizaje, no por “parche” en entrada.

Cuándo usarla:
- cuando quieres robustez más general para escenarios reales.

### Criterio rápido de decisión

- **Poco tiempo / pocos recursos:** centrar antes de predecir.
- **Más tiempo de entrenamiento:** augmentation para que la red aprenda invariancia posicional.
