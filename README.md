---
title: Scoliosis Detection Demo
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Detección de Escoliosis — Demo

Demo web para detección de escoliosis a partir de una fotografía de la espalda,
desarrollada como parte de un proyecto de tesis. Combina tres modelos:

- **E1 — CNN (EfficientNetB0):** clasifica la imagen completa e incluye un mapa
  de calor Grad-CAM con las zonas que más influyeron en la predicción.
- **E2 — Pose + XGBoost:** extrae 12 features geométricas a partir de los
  landmarks de pose (MediaPipe) y clasifica con un modelo XGBoost.
- **E3 — Híbrido (CNN + Pose):** combina las representaciones de imagen y de
  pose en una sola red.

El diagnóstico final se obtiene por **consenso de mayoría simple** entre E1,
E2 y E3.

## Ejecutar localmente

### Con Docker
```bash
docker build -t scoliosis-demo .
docker run -p 7860:7860 scoliosis-demo
# Abrir http://localhost:7860
```

### Sin Docker
```bash
pip install -r requirements.txt
python -m uvicorn webapp.app:app --reload --host 0.0.0.0 --port 8000
# Abrir http://localhost:8000
```

## Limitaciones conocidas

- E2/E3 requieren que MediaPipe detecte los 8 landmarks de pose (hombros,
  codos, muñecas, caderas). Si la imagen no muestra el cuerpo completo,
  ambos modelos devuelven `N/A`.
- E3 puede mostrar overconfianza (probabilidades cercanas a 0% o 100%) en
  imágenes fuera de la distribución de entrenamiento. El consenso de mayoría
  mitiga el impacto de estos casos.
