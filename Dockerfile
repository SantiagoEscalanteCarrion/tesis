FROM python:3.10-slim

# Dependencias de sistema requeridas por opencv-python y mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente
COPY *.py ./
COPY webapp/ webapp/

# Modelos entrenados y artefacto de MediaPipe (evita descarga en runtime)
COPY outputs/ outputs/
COPY pose_landmarker_full.task .

EXPOSE 7860

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "7860"]
