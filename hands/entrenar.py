# hands/entrenar.py - Versión actualizada
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import os
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import logging

router = APIRouter()

DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request models
class TrainingRequest(BaseModel):
    model_name: str
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001

class ModelSelectionRequest(BaseModel):
    category: str
    model_name: str

# Estado global de entrenamiento
class TrainingStatus:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.message = ""
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.current_category = ""
        self.current_model = ""

training_status = TrainingStatus()

def preprocess_data(X: np.ndarray, y: np.ndarray, category: str, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesa los datos para mejorar el entrenamiento"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar el scaler con nombre específico del modelo
    scaler_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled, y

def create_advanced_model(input_shape: int, num_classes: int, learning_rate: float = 0.001) -> Sequential:
    """Crea un modelo más avanzado con parámetros configurables"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_process_category_data(category: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga y procesa datos de una categoría"""
    file_path = os.path.join(DATA_DIR, f"Category.{category}.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Datos de categoría '{category}' no encontrados")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X, y = [], []
    labels = list(data.keys())

    insufficient_labels = []
    for label, samples in data.items():
        if len(samples) < 20:
            insufficient_labels.append(f"{label}: {len(samples)}/20")

    if insufficient_labels:
        raise ValueError(f"Etiquetas insuficientes: {', '.join(insufficient_labels)}")

    for label, samples in data.items():
        for sample in samples:
            if isinstance(sample, dict) and 'landmarks' in sample:
                landmarks = sample['landmarks']
            else:
                landmarks = sample

            if len(landmarks) == 126:
                X.append(landmarks)
                y.append(label)
            else:
                logger.warning(f"Muestra inválida en {label}: {len(landmarks)} landmarks")

    if len(X) == 0:
        raise ValueError("No se encontraron muestras válidas")

    logger.info(f"Cargadas {len(X)} muestras para {len(labels)} etiquetas")
    return np.array(X), np.array(y), labels

@router.post("/{category}/advanced")
async def train_advanced_model(category: str, request: TrainingRequest, background_tasks: BackgroundTasks):
    """Entrena un modelo avanzado con nombre específico"""
    if training_status.status == "training":
        return JSONResponse(
            status_code=409,
            content={"error": "Ya hay un entrenamiento en progreso"}
        )

    background_tasks.add_task(
        train_model_background, 
        category, 
        request.model_name, 
        request.epochs,
        request.batch_size,
        request.learning_rate
    )

    return JSONResponse({
        "message": f"Entrenamiento de '{category}/{request.model_name}' iniciado en segundo plano",
        "model_name": request.model_name,
        "category": category,
        "epochs": request.epochs,
        "status": "started",
        "check_progress_url": f"/train/progress/{category}"
    })

def train_model_background(category: str, model_name: str, epochs: int, batch_size: int, learning_rate: float):
    """Función de entrenamiento en segundo plano con soporte para múltiples modelos"""
    global training_status
    try:
        training_status.status = "training"
        training_status.progress = 0
        training_status.message = f"Iniciando entrenamiento '{model_name}' para categoría '{category}'..."
        training_status.start_time = datetime.now()
        training_status.current_category = category
        training_status.current_model = model_name

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        # Cargar datos
        training_status.message = "Cargando datos..."
        training_status.progress = 10
        X, y, labels = load_and_process_category_data(category)

        # Preprocesar
        training_status.message = "Preprocesando datos..."
        training_status.progress = 20
        X_processed, y_processed = preprocess_data(X, y, category, model_name)

        # Codificar etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_processed)

        # División de datos
        training_status.message = "Dividiendo datos..."
        training_status.progress = 30
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        # Crear modelo
        training_status.message = "Creando modelo..."
        training_status.progress = 40
        model = create_advanced_model(X_processed.shape[1], len(labels), learning_rate)

        callbacks = []

        # Entrenamiento
        training_status.message = "Entrenando modelo..."
        training_status.progress = 50
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        training_status.progress = 80
        training_status.message = "Evaluando modelo..."
        val_predictions = model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)

        accuracy = accuracy_score(y_val, val_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_pred_classes, average='weighted'
        )

        # Guardar modelo con nombre específico
        training_status.message = "Guardando modelo..."
        training_status.progress = 90
        
        model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
        model.save(model_path)

        # Guardar encoder específico del modelo
        encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)

        # Guardar información completa del modelo
        model_info = {
            "category": category,
            "model_name": model_name,
            "labels": labels,
            "num_samples": len(X),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_date": datetime.now().isoformat(),
            "final_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "val_loss": float(min(history.history['val_loss']))
            },
            "training_history": {
                "total_epochs": len(history.history['loss']),
                "best_epoch": int(np.argmin(history.history['val_loss'])) + 1,
                "final_train_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1])
            },
            "model_files": {
                "model_path": model_path,
                "encoder_path": encoder_path,
                "scaler_path": os.path.join(MODELS_DIR, f"{category}_{model_name}_scaler.pkl")
            }
        }
        
        info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        # Completar entrenamiento
        training_status.status = "completed"
        training_status.progress = 100
        training_status.message = f"Entrenamiento '{model_name}' completado exitosamente"
        training_status.end_time = datetime.now()
        training_status.metrics = model_info["final_metrics"]

        logger.info(f"Entrenamiento completado para {category}/{model_name} con {accuracy:.2%} de precisión")

    except Exception as e:
        training_status.status = "error"
        training_status.message = f"Error durante entrenamiento: {str(e)}"
        training_status.end_time = datetime.now()
        logger.error(f"Error en entrenamiento: {e}")

@router.get("/progress/{category}")
def get_training_progress(category: str):
    """Obtiene el progreso actual del entrenamiento"""
    duration = None
    if training_status.start_time:
        end_time = training_status.end_time or datetime.now()
        duration = int((end_time - training_status.start_time).total_seconds())

    return JSONResponse({
        "status": training_status.status,
        "progress": training_status.progress,
        "message": training_status.message,
        "duration_seconds": duration,
        "metrics": training_status.metrics,
        "current_category": training_status.current_category,
        "current_model": training_status.current_model
    })

@router.get("/{category}/models")
def list_category_models(category: str):
    """Lista todos los modelos disponibles para una categoría"""
    if not os.path.exists(MODELS_DIR):
        return JSONResponse({
            "category": category,
            "models": [],
            "total": 0
        })

    models = []
    
    # Buscar archivos de información de modelos para la categoría específica
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith(f"{category}_") and filename.endswith("_info.json"):
            try:
                with open(os.path.join(MODELS_DIR, filename), "r", encoding="utf-8") as f:
                    model_info = json.load(f)
                
                # Verificar que los archivos del modelo existan
                model_name = model_info.get("model_name", "unknown")
                model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
                encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
                
                model_data = {
                    **model_info,
                    "model_exists": os.path.exists(model_path),
                    "encoder_exists": os.path.exists(encoder_path),
                    "ready_for_prediction": os.path.exists(model_path) and os.path.exists(encoder_path),
                    "file_size_mb": round(os.path.getsize(model_path) / (1024*1024), 2) if os.path.exists(model_path) else 0
                }
                
                models.append(model_data)
                
            except Exception as e:
                logger.error(f"Error leyendo modelo {filename}: {e}")

    # Ordenar por fecha de entrenamiento (más reciente primero)
    models.sort(key=lambda x: x.get("training_date", ""), reverse=True)

    return JSONResponse({
        "category": category,
        "models": models,
        "total": len(models)
    })

@router.get("/models/all")
def list_all_models():
    """Lista todos los modelos disponibles organizados por categoría"""
    if not os.path.exists(MODELS_DIR):
        return JSONResponse({
            "categories": {},
            "total_models": 0
        })

    categories = {}
    total_count = 0
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith("_info.json"):
            try:
                with open(os.path.join(MODELS_DIR, filename), "r", encoding="utf-8") as f:
                    model_info = json.load(f)
                
                category = model_info.get("category", "unknown")
                model_name = model_info.get("model_name", "unknown")
                
                # Verificar archivos
                model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
                encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
                
                if category not in categories:
                    categories[category] = []
                
                model_data = {
                    **model_info,
                    "ready_for_prediction": os.path.exists(model_path) and os.path.exists(encoder_path),
                    "file_size_mb": round(os.path.getsize(model_path) / (1024*1024), 2) if os.path.exists(model_path) else 0
                }
                
                categories[category].append(model_data)
                total_count += 1
                
            except Exception as e:
                logger.error(f"Error leyendo modelo {filename}: {e}")

    # Ordenar modelos dentro de cada categoría
    for category in categories:
        categories[category].sort(key=lambda x: x.get("training_date", ""), reverse=True)

    return JSONResponse({
        "categories": categories,
        "total_models": total_count,
        "total_categories": len(categories)
    })

@router.delete("/{category}/models/{model_name}")
def delete_model(category: str, model_name: str):
    """Elimina un modelo específico"""
    files_to_delete = [
        f"{category}_{model_name}_model.h5",
        f"{category}_{model_name}_encoder.pkl",
        f"{category}_{model_name}_scaler.pkl",
        f"{category}_{model_name}_info.json"
    ]
    
    deleted_files = []
    
    for filename in files_to_delete:
        file_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(filename)
            except Exception as e:
                logger.error(f"Error eliminando {filename}: {e}")
    
    if deleted_files:
        return JSONResponse({
            "message": f"Modelo '{category}/{model_name}' eliminado exitosamente",
            "deleted_files": deleted_files,
            "total_deleted": len(deleted_files)
        })
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"No se encontró el modelo '{category}/{model_name}'"}
        )

@router.get("/{category}/models/{model_name}/info")
def get_model_info(category: str, model_name: str):
    """Obtiene información detallada de un modelo específico"""
    info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
    
    if not os.path.exists(info_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"No se encontró información del modelo '{category}/{model_name}'"}
        )
    
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        
        # Verificar archivos del modelo
        model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
        encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_scaler.pkl")
        
        # Agregar información de archivos
        file_info = {
            "model_exists": os.path.exists(model_path),
            "encoder_exists": os.path.exists(encoder_path),
            "scaler_exists": os.path.exists(scaler_path),
            "ready_for_prediction": os.path.exists(model_path) and os.path.exists(encoder_path),
            "model_size_mb": round(os.path.getsize(model_path) / (1024*1024), 2) if os.path.exists(model_path) else 0
        }
        
        detailed_info = {
            **model_info,
            "file_info": file_info
        }
        
        return JSONResponse(detailed_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error leyendo información del modelo: {str(e)}"}
        )
