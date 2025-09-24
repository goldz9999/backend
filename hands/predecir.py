from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import os
import cv2
import mediapipe as mp
import pickle
from typing import Optional, Dict, List

router = APIRouter()

MODELS_DIR = "models"

# Inicializar MediaPipe
mp_hands = mp.solutions.hands

# Cache para modelos cargados
loaded_models: Dict[str, Dict] = {}

def extract_landmarks_for_prediction(image_data: bytes) -> Optional[List[float]]:
    """Extrae landmarks de las manos para predicción"""
    
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Si solo hay una mano, rellenar con ceros
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * 63)
            
            return landmarks
        else:
            return None

def load_model(category: str) -> Dict:
    """Carga un modelo y su encoder"""
    
    if category in loaded_models:
        return loaded_models[category]
    
    model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
    encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
    info_path = os.path.join(MODELS_DIR, f"{category}_info.json")
    
    if not all(os.path.exists(path) for path in [model_path, encoder_path, info_path]):
        raise FileNotFoundError(f"Modelo para la categoría '{category}' no encontrado")
    
    # Cargar modelo
    model = tf.keras.models.load_model(model_path)
    
    # Cargar encoder
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Cargar información
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    model_data = {
        "model": model,
        "encoder": encoder,
        "info": info
    }
    
    # Cachear modelo
    loaded_models[category] = model_data
    
    return model_data

@router.post("/{category}")
async def predict_sign(
    category: str,
    image: UploadFile = File(...)
):
    """Predice una seña desde una imagen"""
    
    try:
        # Verificar que el archivo sea una imagen
        if not image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "El archivo debe ser una imagen"}
            )
        
        # Cargar modelo
        model_data = load_model(category)
        model = model_data["model"]
        encoder = model_data["encoder"]
        info = model_data["info"]
        
        # Leer imagen
        image_data = await image.read()
        
        # Extraer landmarks
        landmarks = extract_landmarks_for_prediction(image_data)
        
        if landmarks is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No se detectaron manos en la imagen"}
            )
        
        # Verificar que los landmarks tengan la forma correcta
        if len(landmarks) != info["input_shape"]:
            return JSONResponse(
                status_code=400,
                content={"error": f"Formato de landmarks incorrecto. Esperado: {info['input_shape']}, Recibido: {len(landmarks)}"}
            )
        
        # Hacer predicción
        landmarks_array = np.array([landmarks])
        predictions = model.predict(landmarks_array)
        
        # Obtener probabilidades y etiquetas
        probabilities = predictions[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(probabilities[predicted_class_idx])
        
        # Crear ranking de predicciones
        ranking = []
        for i, prob in enumerate(probabilities):
            label = encoder.inverse_transform([i])[0]
            ranking.append({
                "label": label,
                "confidence": float(prob),
                "percentage": round(float(prob) * 100, 2)
            })
        
        # Ordenar por confianza
        ranking.sort(key=lambda x: x["confidence"], reverse=True)
        
        return JSONResponse({
            "category": category,
            "prediction": predicted_class,
            "confidence": confidence,
            "percentage": round(confidence * 100, 2),
            "ranking": ranking,
            "success": True,
            "hands_detected": True
        })
        
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error durante la predicción: {str(e)}"}
        )

@router.post("/batch/{category}")
async def predict_batch(
    category: str,
    images: List[UploadFile] = File(...)
):
    """Predice múltiples imágenes a la vez"""
    
    if len(images) > 10:  # Limitar a 10 imágenes por batch
        return JSONResponse(
            status_code=400,
            content={"error": "Máximo 10 imágenes por lote"}
        )
    
    try:
        model_data = load_model(category)
        model = model_data["model"]
        encoder = model_data["encoder"]
        info = model_data["info"]
        
        results = []
        
        for i, image in enumerate(images):
            try:
                if not image.content_type.startswith('image/'):
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "error": "No es una imagen válida"
                    })
                    continue
                
                image_data = await image.read()
                landmarks = extract_landmarks_for_prediction(image_data)
                
                if landmarks is None:
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "error": "No se detectaron manos"
                    })
                    continue
                
                if len(landmarks) != info["input_shape"]:
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "error": "Formato de landmarks incorrecto"
                    })
                    continue
                
                # Predicción
                landmarks_array = np.array([landmarks])
                predictions = model.predict(landmarks_array, verbose=0)
                
                probabilities = predictions[0]
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = encoder.inverse_transform([predicted_class_idx])[0]
                confidence = float(probabilities[predicted_class_idx])
                
                results.append({
                    "image_index": i,
                    "filename": image.filename,
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "percentage": round(confidence * 100, 2),
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "image_index": i,
                    "filename": image.filename,
                    "error": f"Error procesando imagen: {str(e)}"
                })
        
        return JSONResponse({
            "category": category,
            "total_images": len(images),
            "results": results,
            "successful_predictions": sum(1 for r in results if r.get("success", False))
        })
        
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error durante predicción batch: {str(e)}"}
        )

@router.get("/available")
def get_available_models():
    """Lista los modelos disponibles para predicción"""
    
    if not os.path.exists(MODELS_DIR):
        return JSONResponse({
            "available_models": [],
            "total": 0
        })
    
    available = []
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith("_info.json"):
            category = filename.replace("_info.json", "")
            
            model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
            encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                with open(os.path.join(MODELS_DIR, filename), "r", encoding="utf-8") as f:
                    info = json.load(f)
                
                available.append({
                    "category": category,
                    "labels": info["labels"],
                    "num_classes": info["num_classes"],
                    "accuracy": round(info["final_accuracy"] * 100, 2),
                    "samples_used": info["num_samples"]
                })
    
    return JSONResponse({
        "available_models": available,
        "total": len(available)
    })

@router.post("/clear-cache")
def clear_model_cache():
    """Limpia la caché de modelos cargados"""
    
    global loaded_models
    cache_size = len(loaded_models)
    loaded_models.clear()
    
    return JSONResponse({
        "message": f"Caché limpiada. {cache_size} modelos removidos de memoria",
        "models_cleared": cache_size
    })