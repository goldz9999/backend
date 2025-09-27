# hands/predecir.py - Versión actualizada con soporte para múltiples modelos
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import os
import pickle
from typing import Dict, List, Optional
from pydantic import BaseModel


router = APIRouter()

MODELS_DIR = "models"

# Cache optimizado para múltiples modelos
model_cache: Dict[str, Dict] = {}

class PredictionRequest(BaseModel):
    landmarks: List[float]  # Array de 126 valores
    confidence_threshold: float = 0.7
    return_all_probabilities: bool = False
    model_name: Optional[str] = None  # Modelo específico a usar

class BatchPredictionRequest(BaseModel):
    landmarks_batch: List[List[float]]
    model_name: Optional[str] = None
    confidence_threshold: float = 0.7

def get_model_cache_key(category: str, model_name: str = "default") -> str:
    """Genera clave única para el cache del modelo"""
    return f"{category}_{model_name}"

def load_model_components(category: str, model_name: str = "default") -> Dict:
    """Carga todos los componentes del modelo de forma optimizada"""
    
    cache_key = get_model_cache_key(category, model_name)
    
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    # Rutas de archivos específicos del modelo
    model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
    encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_scaler.pkl")
    info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
    
    # Verificar archivos necesarios
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo '{category}/{model_name}' no encontrado")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder del modelo '{category}/{model_name}' no encontrado")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Información del modelo '{category}/{model_name}' no encontrada")
    
    try:
        # Cargar modelo
        model = tf.keras.models.load_model(model_path)
        
        # Cargar encoder
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        
        # Cargar scaler si existe
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        
        # Cargar información
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        
        # Cachear componentes
        components = {
            "model": model,
            "encoder": encoder,
            "scaler": scaler,
            "info": info
        }
        
        model_cache[cache_key] = components
        return components
        
    except Exception as e:
        raise Exception(f"Error cargando modelo '{category}/{model_name}': {str(e)}")

def get_available_models_for_category(category: str) -> List[str]:
    """Obtiene lista de modelos disponibles para una categoría"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith(f"{category}_") and filename.endswith("_model.h5"):
            # Extraer el nombre del modelo
            model_name = filename.replace(f"{category}_", "").replace("_model.h5", "")
            models.append(model_name)
    
    return models

def get_default_model_for_category(category: str) -> str:
    """Obtiene el modelo por defecto para una categoría (el más reciente)"""
    available_models = get_available_models_for_category(category)
    
    if not available_models:
        raise FileNotFoundError(f"No hay modelos disponibles para la categoría '{category}'")
    
    # Si hay un modelo llamado "default", usarlo
    if "default" in available_models:
        return "default"
    
    # Sino, encontrar el más reciente basado en fecha de modificación
    latest_model = None
    latest_time = 0
    
    for model_name in available_models:
        model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
        if os.path.exists(model_path):
            mod_time = os.path.getmtime(model_path)
            if mod_time > latest_time:
                latest_time = mod_time
                latest_model = model_name
    
    return latest_model or available_models[0]

@router.post("/{category}/predict")
async def predict_landmarks(category: str, request: PredictionRequest):
    """Predicción avanzada con landmarks procesados y selección de modelo"""
    
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
        
        # Determinar qué modelo usar
        model_name = request.model_name
        if not model_name:
            try:
                model_name = get_default_model_for_category(category)
            except FileNotFoundError as e:
                return JSONResponse(
                    status_code=404,
                    content={"error": str(e)}
                )
        
        # Cargar modelo
        try:
            components = load_model_components(category, model_name)
        except FileNotFoundError as e:
            return JSONResponse(
                status_code=404,
                content={"error": str(e)}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
        
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
            "model_name": model_name,
            "prediction": predicted_class,
            "confidence": confidence,
            "percentage": round(confidence * 100, 2),
            "high_confidence": high_confidence,
            "model_info": {
                "training_date": info.get("training_date"),
                "accuracy": info.get("final_metrics", {}).get("accuracy", 0),
                "total_samples": info.get("num_samples", 0)
            }
        }
        
        # Incluir todas las probabilidades si se solicita
        if request.return_all_probabilities:
            result["all_predictions"] = ranking
        else:
            result["top_3"] = ranking[:3]
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en predicción: {str(e)}"}
        )

@router.post("/{category}/batch-predict")
async def predict_batch_landmarks(category: str, request: BatchPredictionRequest):
    """Predicción en lote para múltiples muestras"""
    
    if len(request.landmarks_batch) > 100:
        return JSONResponse(
            status_code=400,
            content={"error": "Máximo 100 muestras por lote"}
        )
    
    try:
        # Determinar modelo a usar
        model_name = request.model_name
        if not model_name:
            model_name = get_default_model_for_category(category)
        
        components = load_model_components(category, model_name)
        model = components["model"]
        encoder = components["encoder"]
        scaler = components["scaler"]
        
        results = []
        
        for i, landmarks in enumerate(request.landmarks_batch):
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
                    "percentage": round(confidence * 100, 2),
                    "high_confidence": confidence >= request.confidence_threshold
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if "prediction" in r)
        
        return JSONResponse({
            "category": category,
            "model_name": model_name,
            "total_samples": len(request.landmarks_batch),
            "successful": successful,
            "failed": len(request.landmarks_batch) - successful,
            "results": results
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en predicción batch: {str(e)}"}
        )

@router.get("/available")
def get_available_models():
    """Lista todos los modelos disponibles organizados por categoría"""
    if not os.path.exists(MODELS_DIR):
        return JSONResponse({
            "available_models": [],
            "categories": {},
            "total": 0
        })
    
    categories = {}
    all_models = []
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith("_info.json"):
            try:
                with open(os.path.join(MODELS_DIR, filename), "r", encoding="utf-8") as f:
                    info = json.load(f)
                
                category = info.get("category", "unknown")
                model_name = info.get("model_name", "unknown")
                
                # Verificar que el modelo existe
                model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
                encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
                
                if os.path.exists(model_path) and os.path.exists(encoder_path):
                    model_data = {
                        "category": category,
                        "model_name": model_name,
                        "labels": info.get("labels", []),
                        "num_classes": len(info.get("labels", [])),
                        "accuracy": round(info.get("final_metrics", {}).get("accuracy", 0) * 100, 2),
                        "samples_used": info.get("num_samples", 0),
                        "training_date": info.get("training_date", ""),
                        "epochs": info.get("epochs", 0),
                        "ready_for_prediction": True
                    }
                    
                    all_models.append(model_data)
                    
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(model_data)
                    
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")
    
    # Ordenar modelos dentro de cada categoría por fecha de entrenamiento
    for category in categories:
        categories[category].sort(key=lambda x: x.get("training_date", ""), reverse=True)
    
    return JSONResponse({
        "available_models": all_models,
        "categories": categories,
        "total": len(all_models)
    })

@router.get("/{category}/models")
def get_category_models(category: str):
    """Obtiene todos los modelos disponibles para una categoría específica"""
    available_models = get_available_models_for_category(category)
    
    if not available_models:
        return JSONResponse({
            "category": category,
            "models": [],
            "total": 0,
            "default_model": None
        })
    
    models_info = []
    
    for model_name in available_models:
        try:
            info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                
                model_data = {
                    "model_name": model_name,
                    "labels": info.get("labels", []),
                    "accuracy": round(info.get("final_metrics", {}).get("accuracy", 0) * 100, 2),
                    "samples_used": info.get("num_samples", 0),
                    "training_date": info.get("training_date", ""),
                    "epochs": info.get("epochs", 0),
                    "ready_for_prediction": True
                }
                models_info.append(model_data)
                
        except Exception as e:
            print(f"Error leyendo info de {model_name}: {e}")
    
    # Ordenar por fecha de entrenamiento
    models_info.sort(key=lambda x: x.get("training_date", ""), reverse=True)
    
    try:
        default_model = get_default_model_for_category(category)
    except:
        default_model = models_info[0]["model_name"] if models_info else None
    
    return JSONResponse({
        "category": category,
        "models": models_info,
        "total": len(models_info),
        "default_model": default_model
    })

@router.get("/{category}/{model_name}/info")
def get_model_detailed_info(category: str, model_name: str):
    """Obtiene información detallada de un modelo específico"""
    
    try:
        components = load_model_components(category, model_name)
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
        
        # Tamaño del modelo
        model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
        model_size = round(os.path.getsize(model_path) / (1024*1024), 2) if os.path.exists(model_path) else 0
        
        detailed_info = {
            **info,
            "model_architecture": architecture,
            "total_parameters": model.count_params(),
            "trainable_parameters": sum(layer.count_params() for layer in model.layers if layer.trainable),
            "model_size_mb": model_size,
            "cache_key": get_model_cache_key(category, model_name)
        }
        
        return JSONResponse(detailed_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error obteniendo info del modelo: {str(e)}"}
        )

@router.post("/load-model")
async def load_specific_model(category: str, model_name: str):
    """Carga un modelo específico en memoria"""
    try:
        components = load_model_components(category, model_name)
        cache_key = get_model_cache_key(category, model_name)
        
        return JSONResponse({
            "message": f"Modelo '{category}/{model_name}' cargado exitosamente",
            "cache_key": cache_key,
            "model_info": {
                "labels": components["info"].get("labels", []),
                "accuracy": components["info"].get("final_metrics", {}).get("accuracy", 0),
                "training_date": components["info"].get("training_date", "")
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error cargando modelo: {str(e)}"}
        )

@router.delete("/cache")
def clear_model_cache():
    """Limpia el cache de modelos cargados"""
    global model_cache
    cache_size = len(model_cache)
    cached_models = list(model_cache.keys())
    model_cache.clear()
    
    return JSONResponse({
        "message": f"Cache limpiado. {cache_size} modelos removidos de memoria",
        "models_cleared": cache_size,
        "cached_models": cached_models
    })

@router.delete("/cache/{category}")
def clear_category_cache(category: str):
    """Limpia el cache de modelos de una categoría específica"""
    global model_cache
    
    keys_to_remove = [key for key in model_cache.keys() if key.startswith(f"{category}_")]
    
    for key in keys_to_remove:
        del model_cache[key]
    
    return JSONResponse({
        "message": f"Cache de categoría '{category}' limpiado. {len(keys_to_remove)} modelos removidos",
        "models_cleared": len(keys_to_remove),
        "cleared_models": keys_to_remove
    })
@router.post("/{category}/practice/check")
def check_practice(category: str, landmarks: List[float] = Query(...), confidence_threshold: float = 0.7):
    """
    Endpoint de práctica: predice la clase usando el modelo entrenado más reciente
    """
    if len(landmarks) != 126:
        return JSONResponse({
            "error": f"Se esperan 126 landmarks, recibidos: {len(landmarks)}",
            "success": False
        })
    
    # Seleccionar modelo por defecto (más reciente)
    try:
        model_name = get_default_model_for_category(category)
    except FileNotFoundError:
        return JSONResponse({
            "error": f"No hay modelos entrenados para la categoría '{category}'",
            "success": False
        })
    
    # Cargar modelo
    try:
        components = load_model_components(category, model_name)
    except Exception as e:
        return JSONResponse({
            "error": f"No se pudo cargar el modelo: {str(e)}",
            "success": False
        })
    
    model = components["model"]
    encoder = components["encoder"]
    scaler = components["scaler"]
    
    # Preparar datos
    landmarks_array = np.array([landmarks])
    if scaler:
        landmarks_array = scaler.transform(landmarks_array)
    
    # Predicción
    predictions = model.predict(landmarks_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_label = encoder.inverse_transform([predicted_idx])[0]
    confidence = float(predictions[predicted_idx])
    
    high_confidence = confidence >= confidence_threshold
    
    # Ranking top 3
    ranking = []
    for i, prob in enumerate(predictions):
        label = encoder.inverse_transform([i])[0]
        ranking.append({
            "label": label,
            "confidence": float(prob),
            "percentage": round(float(prob) * 100, 2)
        })
    ranking.sort(key=lambda x: x["confidence"], reverse=True)
    
    return JSONResponse({
        "success": True,
        "category": category,
        "model_name": model_name,
        "prediction": predicted_label,
        "confidence": confidence,
        "high_confidence": high_confidence,
        "top_3": ranking[:3]
    })