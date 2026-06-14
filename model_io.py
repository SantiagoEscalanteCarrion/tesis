"""
model_io.py — Carga de modelos .keras con compatibilidad entre versiones de Keras.

Los modelos de este proyecto se entrenaron y guardaron con Keras 3.13.2, que
serializa las capas Dense incluyendo un campo "quantization_config". Si el
entorno de despliegue resuelve una versión de Keras 3 anterior (que no
reconoce ese campo), `keras.models.load_model` falla con:

    TypeError: ... Unrecognized keyword arguments passed to Dense:
    {'quantization_config': None}

`load_model_compat` reintenta la carga eliminando ese campo (y el
"compile_config", innecesario para inferencia) directamente del config.json
embebido en el archivo .keras, y carga los pesos por separado.
"""

import json
import tempfile
import zipfile

from tensorflow import keras


def _strip_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            _strip_quantization_config(value)
    elif isinstance(obj, list):
        for value in obj:
            _strip_quantization_config(value)
    return obj


def load_model_compat(path):
    """Carga un modelo .keras, con fallback si la versión de Keras instalada
    no soporta 'quantization_config' en las configs de capas Dense."""
    try:
        return keras.models.load_model(path, compile=False)
    except TypeError as e:
        if "quantization_config" not in str(e):
            raise
        print(f"[model_io] {path}: 'quantization_config' no soportado, "
              f"reconstruyendo modelo sin ese campo...")

    with zipfile.ZipFile(path) as zf:
        config = json.loads(zf.read("config.json"))
        config.pop("compile_config", None)
        _strip_quantization_config(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = zf.extract("model.weights.h5", tmpdir)
            model = keras.models.model_from_json(json.dumps(config))
            model.load_weights(weights_path)
    return model
