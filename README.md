# TFM · Predicción de Consumo y Precio Eléctrico — Madrid

Sistema de predicción de precio y demanda eléctrica para las próximas 48 horas,
combinando datos de REE, OMIE y Open-Meteo con modelos ML/DL.

---

## Estructura del proyecto

```
tfm-energia/
├── data/
│   ├── raw/                    # Datos crudos descargados (no subir a Git)
│   ├── processed/              # Datasets procesados
│   └── models/                 # Modelos entrenados y scalers
│       └── nf/                 # Modelos neuralforecast (N-HiTS, TFT)
├── notebooks/
│   ├── 01_extraccion_datos.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelado.ipynb
│   └── 05_modelos_avanzados.ipynb
├── app/
│   ├── app.py                  # App Streamlit principal
│   └── pages/
│       ├── 1_dashboard.py
│       ├── 2_simulador.py
│       └── 3_modelos.py
├── predict.py                  # Motor de inferencia
├── requirements.txt
├── .env                        # API keys (NO subir a Git)
├── .env.example                # Plantilla de API keys
└── .gitignore
```

---

## Instalación en VSCode

### 1. Clonar / crear la carpeta del proyecto

```bash
mkdir tfm-energia
cd tfm-energia
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ Si tienes GPU NVIDIA, instala PyTorch con CUDA en lugar del de requirements:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### 4. Configurar API keys

```bash
cp .env.example .env
```

Abre `.env` y rellena:
- `ESIOS_API_KEY` → tu token de e.sios (REE)
- `AEMET_API_KEY` → tu API key de AEMET (opcional)

### 5. Descomprimir modelos

Descomprime `tfm_modelos_export.zip` en la raíz del proyecto.
Debe quedar la carpeta `data/` con los modelos dentro.

### 6. Verificar instalación

```bash
python predict.py --modelo_precio nhits --modelo_demanda xgboost
```

Si todo va bien verás las predicciones de las próximas 48h.

---

## Ejecutar la app Streamlit

```bash
streamlit run app/app.py
```

La app se abrirá en `http://localhost:8501`

---

## Uso del motor de inferencia

```python
from predict import Predictor

# Modelos disponibles: 'xgboost', 'lstm', 'nhits', 'tft'
p = Predictor(modelo_precio='nhits', modelo_demanda='xgboost')
resultado = p.predecir()

print(resultado['hora_barata'])   # '14:00'
print(resultado['precio_min'])    # 45.2 €/MWh
print(resultado['precio'])        # lista de 48 valores
```

---

## Modelos en producción

| Variable | Modelo | MAE |
|----------|--------|-----|
| Precio   | N-HiTS | ver resultados_finales.json |
| Demanda  | XGBoost | 8.799 MW |

---

## Fuentes de datos

| Fuente | Dato | Auth |
|--------|------|------|
| REE / e.sios | Demanda horaria Madrid | Token |
| OMIE | Precio spot €/MWh | Sin auth |
| Open-Meteo | Meteorología horaria | Sin auth |
| holidays (Python) | Festivos España/Madrid | Sin auth |

## IMPORTANTE

Los modelos entrenados no se han subido, por lo que es necesario ejecutar los notebooks antes de usar la app.