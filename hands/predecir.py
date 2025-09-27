from fastapi import APIRouter
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import os
import pickle
from typing import Dict, List
from pydantic import BaseModel

router = APIRouter()

MODELS_DIR = "models"

# Cache optimizado para modelos
model_cache: Dict[str, Dict] = {}

class PredictionRequest(BaseModel):
    landmarks: List[float]  # Array de 126 valores
    confidence_threshold: float = 0.7
    return_all_probabilities: bool = False

def load_model_components(category: str) -> Dict:
    """Carga todos los componentes del modelo de forma optimizada"""
    
    if category in model_cache:
        return model_cache[category]
    
    # Verificar archivos necesarios
    model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
    encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    info_path = os.path.join(MODELS_DIR, f"{category}_info.json")
    
    if not all(os.path.exists(path) for path in [model_path, encoder_path, info_path]):
        raise FileNotFoundError(f"Modelo '{category}' incompleto o no encontrado")
    
    # Cargar componentes
    model = tf.keras.models.load_model(model_path)
    
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    
    # Cargar scaler si existe
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    # Cachear componentes
    components = {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "info": info
    }
    
    model_cache[category] = components
    return components

@router.post("/{category}/predict")
async def predict_landmarks(category: str, request: PredictionRequest):
    """Predicción avanzada con landmarks procesados"""
    
    try:
        # Validar landmarks
        if len(request.landmarks) != 126:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Se esperan 126 landmarks, recibidos: {len(request.landmarks)}",
                    "expected_format": "2 manos × 21 landmarks × 3 coordenadas = 126 valores"
                }
            )
        
        # Cargar modelo
        components = load_model_components(category)
        model = components["model"]
        encoder = components["encoder"]
        scaler = components["scaler"]
        info = components["info"]
        
        # Preparar datos
        landmarks_array = np.array([request.landmarks])
        
        # Aplicar escalado si existe
        if scaler:
            landmarks_array = scaler.transform(landmarks_array)
        
        # Hacer predicción
        predictions = model.predict(landmarks_array, verbose=0)
        probabilities = predictions[0]
        
        # Obtener predicción principal
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(probabilities[predicted_class_idx])
        
        # Verificar umbral de confianza
        high_confidence = confidence >= request.confidence_threshold
        
        # Crear ranking de todas las predicciones
        ranking = []
        for i, prob in enumerate(probabilities):
            label = encoder.inverse_transform([i])[0]
            ranking.append({
                "label": label,
                "confidence": float(prob),
                "percentage": round(float(prob) * 100, 2)
            })
        
        ranking.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Respuesta base
        result = {
            "category": category,
            "prediction": predicted_class,
            "confidence": confidence,
            "percentage": round(confidence * 100, 2),
            "high_confidence": high_confidence,
            "model_info": {
                "training_date": info.get("training_date"),
                "accuracy": info.get("final_metrics", {}).get("accuracy", 0)
            }
        }
        
        # Incluir todas las probabilidades si se solicita
        if request.return_all_probabilities:
            result["all_predictions"] = ranking
        else:
            result["top_3"] = ranking[:3]
        
        return JSONResponse(result)
        
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en predicción: {str(e)}"}
        )

@router.post("/{category}/batch-predict")
async def predict_batch_landmarks(category: str, landmarks_batch: List[List[float]]):
    """Predicción en lote para múltiples muestras"""
    
    if len(landmarks_batch) > 100:
        return JSONResponse(
            status_code=400,
            content={"error": "Máximo 100 muestras por lote"}
        )
    
    try:
        components = load_model_components(category)
        model = components["model"]
        encoder = components["encoder"]
        scaler = components["scaler"]
        
        results = []
        
        for i, landmarks in enumerate(landmarks_batch):
            if len(landmarks) != 126:
                results.append({
                    "index": i,
                    "error": f"Landmarks inválidos: {len(landmarks)}/126"
                })
                continue
            
            try:
                landmarks_array = np.array([landmarks])
                
                if scaler:
                    landmarks_array = scaler.transform(landmarks_array)
                
                predictions = model.predict(landmarks_array, verbose=0)
                probabilities = predictions[0]
                
                predicted_idx = np.argmax(probabilities)
                predicted_label = encoder.inverse_transform([predicted_idx])[0]
                confidence = float(probabilities[predicted_idx])
                
                results.append({
                    "index": i,
                    "prediction": predicted_label,
                    "confidence": confidence,
                    "percentage": round(confidence * 100, 2)
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if "prediction" in r)
        
        return JSONResponse({
            "total_samples": len(landmarks_batch),
            "successful": successful,
            "failed": len(landmarks_batch) - successful,
            "results": results
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en predicción batch: {str(e)}"}
        )

@router.get("/{category}/model-info")
def get_model_detailed_info(category: str):
    """Obtiene información detallada del modelo"""
    
    try:
        components = load_model_components(category)
        info = components["info"]
        model = components["model"]
        
        # Información de la arquitectura
        architecture = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                "index": i,
                "name": layer.name,
                "type": type(layer).__name__,
                "output_shape": str(layer.output_shape),
                "params": layer.count_params()
            }
            architecture.append(layer_info)
        
        detailed_info = {
            **info,
            "model_architecture": architecture,
            "total_parameters": model.count_params(),
            "trainable_parameters": sum(layer.count_params() for layer in model.layers if layer.trainable),
            "model_size_mb": round(os.path.getsize(os.path.join(MODELS_DIR, f"{category}_model.h5")) / (1024*1024), 2)
        }
        
        return JSONResponse(detailed_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error obteniendo info del modelo: {str(e)}"}
        )

@router.delete("/cache")
def clear_model_cache():
    """Limpia el cache de modelos cargados"""
    global model_cache
    cache_size = len(model_cache)
    model_cache.clear()
    
    return JSONResponse({
        "message": f"Cache limpiado. {cache_size} modelos removidos de memoria",
        "models_cleared": cache_size
    })
