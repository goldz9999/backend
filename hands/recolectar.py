from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from enum import Enum
import json, os

router = APIRouter()

DATA_DIR = "data"

# Definimos las categorías posibles
class Category(str, Enum):
    vocales = "vocales"
    numeros = "numeros"
    operaciones = "operaciones"
    palabras = "palabras"

@router.post("/sample")
def collect_sample(category: Category, label: str, landmarks: list = Body(...)):
    os.makedirs(DATA_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, f"{category}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    if label not in data:
        data[label] = []

    # Guardamos la lista de landmarks como una muestra
    data[label].append(landmarks)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return JSONResponse({
        "message": f"Muestra guardada en categoría '{category}' con etiqueta '{label}'",
        "total": len(data[label])
    })
