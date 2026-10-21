# Digit Detector App (Streamlit + scikit-learn)

Aplicación educativa para reconocer **números escritos (0..9)** con:

- **Árbol de Decisión**
- **Random Forest**
- **Red Neuronal MLP pequeña**
- **Red Neuronal MLP profunda (regularizada)**
- Interfaz Streamlit con **lienzo de dibujo** (`streamlit-drawable-canvas`)

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
└── dataset/            # opcional: si quieres entrenar desde ruta local
```

---

## 2) Dataset usado (Kaggle)

El entrenamiento usa por defecto este dataset:

```python
import kagglehub
path = kagglehub.dataset_download("karnikakapoor/digits")
print(path)
```

El script `train.py` ya hace esta descarga automáticamente cuando no pasas `--dataset-root`.

---

## 3) Formato del dataset (si usas ruta local)

El entrenamiento espera una carpeta `dataset` con subcarpetas por clase:

```text
dataset/
├── 0/
│   ├── img_001.png
│   └── ...
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
```

Cada subcarpeta contiene imágenes del dígito correspondiente sobre fondo blanco.

## 4) Entorno virtual e instalación (Windows PowerShell)

```powershell
cd c:\Users\abel.perez\Desktop\ML\SignDetectorApp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si PowerShell bloquea el activate:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

## 5) Entrenamiento del modelo

Entrenamiento estándar (descarga automática de Kaggle `karnikakapoor/digits`):

```powershell
python train.py
```

Si prefieres una ruta local concreta:

```powershell
python train.py --dataset-root "C:\ruta\al\dataset"
```

Opciones útiles:

```powershell
# reducir tamaño de entrada (más ligero para Render)
python train.py --image-size 28

# limitar imágenes por clase (0 = todas)
python train.py --max-images-per-class 500
```

Esto generará/actualizará `modelo.joblib`.

---

## 6) Ejecución local

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Abrir `http://localhost:8501`.

La app permite:
- Dibujar un número en un lienzo
- Subir una imagen
- Seleccionar qué modelos ejecutar (1/4, 2/4, 3/4 o 4/4)
- Comparar predicción y confianza entre modelos
- Ver voto mayoritario del Random Forest y probabilidades por clase en todos los modelos

---

## 7) Troubleshooting rápido

Si ves este error:

`ModuleNotFoundError: No module named 'streamlit_drawable_canvas'`

ejecuta en la carpeta del proyecto:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Esto fuerza a usar el entorno correcto y evita mezclar el `streamlit` global del sistema.

---

## 8) Deploy en Render

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

> Importante: Render no entrena automáticamente con este build.
> Debes subir `modelo.joblib` ya entrenado al repositorio (o añadir entrenamiento al build, si lo prefieres).

## 9) Notas de rendimiento

- El pipeline redimensiona imágenes para mantener el modelo pequeño y estable.
- Evita usar imágenes gigantes para entrenar; 28x28 o 32x32 suele ser suficiente en este enfoque clásico.
- Random Forest no modela geometría espacial como una CNN; funciona mejor con preprocesado simple y consistente.
- Las MLP también reciben imagen aplanada (no convolucional), por lo que se mantienen simples y rápidas para aprendizaje inicial.
