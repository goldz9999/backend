from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from enum import Enum
import json
import os
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional

router = APIRouter()

DATA_DIR = "data"
SAMPLES_PER_LABEL = 30

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class Category(str, Enum):
    vocales = "vocales"
    numeros = "numeros"
    operaciones = "operaciones"
    palabras = "palabras"

def extract_landmarks(image_data: bytes) -> Optional[List[float]]:
    """Extrae landmarks de las manos desde una imagen"""
    
    # Convertir bytes a imagen
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return None
    
    # Convertir BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar con MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            
            # Extraer landmarks de todas las manos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Si solo hay una mano, rellenar con ceros para la segunda mano
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * 63)  # 21 landmarks * 3 coordenadas
            
            return landmarks
        else:
            return None

@router.post("/sample")
async def collect_sample(
    category: Category = Form(...),
    label: str = Form(...),
    image: UploadFile = File(...)
):
    """Recolecta una muestra desde una imagen"""
    
    if not image.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "El archivo debe ser una imagen"}
        )
    
    # Leer datos de la imagen
    image_data = await image.read()
    
    # Extraer landmarks
    landmarks = extract_landmarks(image_data)
    
    if landmarks is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No se detectaron manos en la imagen"}
        )
    
    # Crear directorio si no existe
    os.makedirs(DATA_DIR, exist_ok=True)
    
    file_path = os.path.join(DATA_DIR, f"Category.{category.value}.json")
    
    # Cargar datos existentes
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    
    # Inicializar label si no existe
    if label not in data:
        data[label] = []
    
    # Verificar si ya se alcanzó el límite de muestras
    if len(data[label]) >= SAMPLES_PER_LABEL:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Ya se alcanzó el límite de {SAMPLES_PER_LABEL} muestras para '{label}'",
                "current_samples": len(data[label])
            }
        )
    
    # Agregar nueva muestra
    data[label].append(landmarks)
    
    # Guardar datos actualizados
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return JSONResponse({
        "message": f"Muestra {len(data[label])}/{SAMPLES_PER_LABEL} guardada para '{label}' en categoría '{category.value}'",
        "current_samples": len(data[label]),
        "max_samples": SAMPLES_PER_LABEL,
        "ready_to_train": len(data[label]) >= SAMPLES_PER_LABEL
    })

@router.get("/status/{category}")
def get_collection_status(category: Category):
    """Obtiene el estado de recolección de una categoría"""
    
    file_path = os.path.join(DATA_DIR, f"Category.{category.value}.json")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "category": category.value,
            "labels": {},
            "total_samples": 0
        })
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    
    # Calcular estado de cada label
    labels_status = {}
    total_samples = 0
    
    for label, samples in data.items():
        sample_count = len(samples)
        labels_status[label] = {
            "samples": sample_count,
            "max_samples": SAMPLES_PER_LABEL,
            "progress": (sample_count / SAMPLES_PER_LABEL) * 100,
            "ready": sample_count >= SAMPLES_PER_LABEL
        }
        total_samples += sample_count
    
    return JSONResponse({
        "category": category.value,
        "labels": labels_status,
        "total_samples": total_samples,
        "total_labels": len(labels_status)
    })

@router.delete("/clear/{category}")
def clear_category_data(category: Category, label: Optional[str] = None):
    """Limpia datos de una categoría o label específico"""
    
    file_path = os.path.join(DATA_DIR, f"Category.{category.value}.json")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "message": f"No hay datos para la categoría '{category.value}'"
        })
    
    if label:
        # Limpiar solo un label específico
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if label in data:
            del data[label]
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            return JSONResponse({
                "message": f"Datos del label '{label}' eliminados de la categoría '{category.value}'"
            })
        else:
            return JSONResponse({
                "message": f"El label '{label}' no existe en la categoría '{category.value}'"
            })
    else:
        # Limpiar toda la categoría
        os.remove(file_path)
        return JSONResponse({
            "message": f"Todos los datos de la categoría '{category.value}' han sido eliminados"
        })