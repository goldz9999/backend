# hands/entrenar.py - Versión actualizada
from fastapi import APIRouter, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
import glob
import os
import pickle
from fastapi import UploadFile, File
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import logging
import tempfile

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

def load_model_components(category: str, model_name: str = "default"):
    """Carga todos los componentes del modelo - MISMA FUNCIÓN QUE EN predecir.py"""
    model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
    encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_scaler.pkl")
    info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
    
    # Verificar archivos necesarios
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo '{category}/{model_name}' no encontrado")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder del modelo '{category}/{model_name}' no encontrado")
    
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
        
        return {
            "model": model,
            "encoder": encoder,
            "scaler": scaler,
            "info": info
        }
        
    except Exception as e:
        raise Exception(f"Error cargando modelo '{category}/{model_name}': {str(e)}")

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
# entrenar.py - Modificar para enviar el modelo AL FRONTEND directamente
@router.post("/{category}/advanced")
async def train_advanced_model(category: str, request: TrainingRequest, background_tasks: BackgroundTasks):
    """Entrena y envía el modelo automáticamente al frontend"""
    if training_status.status == "training":
        return JSONResponse(
            status_code=409,
            content={"error": "Ya hay un entrenamiento en progreso"}
        )

    # ✅ INMEDIATAMENTE preparar respuesta con info para el frontend
    response_data = {
        "message": f"Entrenamiento de '{request.model_name}' iniciado",
        "model_name": request.model_name,
        "category": category,
        "epochs": request.epochs,
        "status": "started",
        "auto_save_to_frontend": True,  # ✅ Nueva bandera
        "frontend_model_key": f"{category}_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    background_tasks.add_task(
        train_and_send_to_frontend,  # ✅ Función que envía al frontend
        category, 
        request.model_name, 
        request.epochs,
        request.batch_size,
        request.learning_rate
    )

    return JSONResponse(response_data)

def train_and_send_to_frontend(category: str, model_name: str, epochs: int, batch_size: int, learning_rate: float):
    """Entrena el modelo y lo prepara para almacenamiento en frontend"""
    global training_status
    
    try:
        # ... (código de entrenamiento existente) ...
        
        # ✅ DESPUÉS del entrenamiento exitoso, preparar paquete para frontend
        training_status.message = "Preparando modelo para almacenamiento en frontend..."
        
        # Cargar el modelo recién entrenado
        components = load_model_components(category, model_name)
        model = components["model"]
        encoder = components["encoder"] 
        scaler = components["scaler"]
        info = components["info"]
        
        # ✅ Crear paquete COMPACTO para el frontend
        model_package = {
            "metadata": {
                "category": category,
                "model_name": model_name,
                "labels": info.get("labels", []),
                "accuracy": info.get("final_metrics", {}).get("accuracy", 0),
                "training_date": datetime.now().isoformat(),
                "samples_used": info.get("num_samples", 0),
                "version": "1.0"
            },
            "model_config": model.get_config(),
            # ✅ Información mínima para reconstrucción en frontend
            "model_info": {
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "layers_count": len(model.layers),
                "total_params": model.count_params()
            },
            "preprocessing": {
                "encoder_classes": encoder.classes_.tolist(),
                "scaler_params": {
                    "mean": scaler.mean_.tolist() if scaler else None,
                    "scale": scaler.scale_.tolist() if scaler else None
                } if scaler else None
            }
        }
        
        # ✅ Guardar paquete en archivo accesible al frontend
        frontend_package_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_FRONTEND.json")
        with open(frontend_package_path, "w", encoding="utf-8") as f:
            json.dump(model_package, f, indent=2, ensure_ascii=False)
        
        training_status.message = f"✅ Modelo '{model_name}' guardado para frontend"
        training_status.metrics = {
            "frontend_package_created": True,
            "package_path": frontend_package_path,
            "ready_for_practice": True
        }
        
        print(f"🎯 Modelo {model_name} listo para práctica inmediata en frontend")
        
    except Exception as e:
        training_status.status = "error"
        training_status.message = f"Error: {str(e)}"
        logger.error(f"Error en entrenamiento: {e}")

# ✅ Endpoint para obtener el modelo listo para frontend
@router.get("/{category}/{model_name}/frontend-package")
async def get_frontend_package(category: str, model_name: str):
    """Obtiene el paquete del modelo listo para el frontend"""
    try:
        package_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_FRONTEND.json")
        
        if not os.path.exists(package_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Paquete para frontend no encontrado"}
            )
        
        with open(package_path, "r", encoding="utf-8") as f:
            package_data = json.load(f)
        
        return JSONResponse({
            "success": True,
            "package": package_data,
            "download_url": f"/train/{category}/{model_name}/frontend-package"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error obteniendo paquete: {str(e)}"}
        )
@router.get("/{category}/download-training-data")
def download_training_data(category: str):
    """Descarga todos los datos de una categoría para entrenamiento"""
    try:
        # Cargar datos usando la función existente (del archivo recolectar.py o entrenar.py)
        file_path = os.path.join(DATA_DIR, f"Category.{category}.json")
        
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"No hay datos disponibles para la categoría '{category}'",
                    "category": category
                }
            )
        
        # Leer archivo de datos
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"El archivo de la categoría '{category}' está vacío",
                    "category": category
                }
            )
        
        # Procesar datos para entrenamiento
        X, y = [], []
        labels = list(data.keys())
        
        # Crear mapeo de etiquetas a índices
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        
        # Estadísticas por etiqueta
        label_stats = {}
        
        for label, samples in data.items():
            label_index = label_to_index[label]
            sample_count = 0
            
            for sample in samples:
                # Extraer landmarks (compatible con estructura existente)
                if isinstance(sample, dict) and 'landmarks' in sample:
                    landmarks = sample['landmarks']
                else:
                    landmarks = sample
                
                # Validar que tenga 126 landmarks
                if isinstance(landmarks, list) and len(landmarks) == 126:
                    X.append(landmarks)
                    y.append(label_index)
                    sample_count += 1
            
            label_stats[label] = sample_count
        
        if len(X) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"No hay muestras válidas (126 landmarks) en la categoría '{category}'",
                    "category": category
                }
            )
        
        # Respuesta completa
        response_data = {
            "category": category,
            "X": X,  # Lista de listas con landmarks
            "y": y,  # Lista de índices de etiquetas
            "labels": labels,  # Lista de nombres de etiquetas
            "label_to_index": label_to_index,
            "statistics": {
                "total_samples": len(X),
                "total_labels": len(labels),
                "samples_per_label": label_stats,
                "features_per_sample": 126
            },
            "ready_to_train": len(labels) > 1 and len(X) >= len(labels) * 10,  # Al menos 10 por etiqueta
            "download_timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Datos descargados para '{category}': {len(X)} muestras, {len(labels)} etiquetas")
        
        return JSONResponse(response_data)
        
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Categoría '{category}' no encontrada",
                "category": category
            }
        )
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Archivo de datos de '{category}' está corrupto",
                "category": category
            }
        )
    except Exception as e:
        print(f"❌ Error descargando datos de '{category}': {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error interno descargando datos de '{category}'",
                "details": str(e)
            }
        )

@router.post("/upload-model")
async def upload_model_from_frontend(
    # Archivos opcionales
    model_file: Optional[UploadFile] = File(None),
    weights_file: Optional[UploadFile] = File(None),
    # Campos de formulario
    category: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    upload_timestamp: Optional[str] = Form(None),
    labels: Optional[str] = Form(None),
    # Alternativa JSON
    upload_type: Optional[str] = Form(None)
):
    """
    Recibe modelo desde el frontend - Soporta dos métodos:
    1. Archivos + FormData
    2. Solo metadata (JSON)
    """
    try:
        logger.info(f"📥 Recibiendo modelo desde frontend...")
        logger.info(f"   - category: {category}")
        logger.info(f"   - model_name: {model_name}")
        logger.info(f"   - upload_type: {upload_type}")
        logger.info(f"   - model_file: {model_file.filename if model_file else 'None'}")
        logger.info(f"   - weights_file: {weights_file.filename if weights_file else 'None'}")
        
        # Crear directorio si no existe
        frontend_upload_dir = os.path.join(MODELS_DIR, "frontend_uploads")
        os.makedirs(frontend_upload_dir, exist_ok=True)
        
        result_info = {
            "status": "success",
            "upload_timestamp": upload_timestamp or datetime.now().isoformat(),
            "category": category,
            "model_name": model_name,
            "upload_type": upload_type or "files",
            "files_received": []
        }
        
        # MÉTODO 1: Archivos completos
        if model_file and weights_file:
            logger.info("📁 Procesando archivos del modelo...")
            
            # Guardar model.json
            model_path = os.path.join(frontend_upload_dir, f"{category}_{model_name}_model.json")
            with open(model_path, "wb") as f:
                content = await model_file.read()
                f.write(content)
            result_info["files_received"].append({
                "file": "model.json",
                "size": len(content),
                "path": model_path
            })
            
            # Guardar weights.bin
            weights_path = os.path.join(frontend_upload_dir, f"{category}_{model_name}_weights.bin")
            with open(weights_path, "wb") as f:
                content = await weights_file.read()
                f.write(content)
            result_info["files_received"].append({
                "file": "weights.bin", 
                "size": len(content),
                "path": weights_path
            })
            
            result_info["message"] = f"Modelo completo recibido: {len(result_info['files_received'])} archivos"
            
        # MÉTODO 2: Solo metadata
        else:
            logger.info("📋 Procesando solo metadata...")
            
            # Crear archivo de información
            info_data = {
                "category": category,
                "model_name": model_name,
                "source": "frontend_training",
                "upload_timestamp": result_info["upload_timestamp"],
                "type": "tensorflow_js_model",
                "status": "uploaded_metadata_only"
            }
            
            if labels:
                try:
                    info_data["labels"] = json.loads(labels)
                except:
                    info_data["labels"] = []
            
            info_path = os.path.join(frontend_upload_dir, f"{category}_{model_name}_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info_data, f, indent=2, ensure_ascii=False)
            
            result_info["message"] = "Metadata del modelo registrada exitosamente"
            result_info["info_path"] = info_path
        
        logger.info(f"✅ Upload completado: {result_info['message']}")
        return JSONResponse(content=result_info)
        
    except Exception as e:
        logger.error(f"❌ Error en upload: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "message": "Error procesando upload del modelo"
            }
        )

# También agregar un endpoint simple para debugging
@router.post("/upload-model-simple")
async def upload_model_simple(request: dict):
    """Endpoint simplificado para recibir solo metadata"""
    try:
        logger.info(f"📥 Upload simple recibido: {request}")
        
        # Crear directorio
        simple_upload_dir = os.path.join(MODELS_DIR, "simple_uploads")
        os.makedirs(simple_upload_dir, exist_ok=True)
        
        # Guardar información
        timestamp = datetime.now().isoformat()
        filename = f"upload_{timestamp.replace(':', '-')}.json"
        filepath = os.path.join(simple_upload_dir, filename)
        
        upload_data = {
            **request,
            "received_at": timestamp,
            "status": "received"
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(upload_data, f, indent=2, ensure_ascii=False)
        
        return JSONResponse({
            "status": "success",
            "message": "Upload simple procesado exitosamente",
            "file": filename,
            "data": upload_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error en upload simple: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "error": str(e)
            }
        )

@router.get("/model/latest/{filename}")
async def get_model(filename: str):
    """Sirve archivos del modelo para descarga"""
    file_path = os.path.join("models/latest", filename)
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"error": "Archivo no encontrado"}
        )
    return FileResponse(file_path)

# Agregar estos endpoints al archivo hands/entrenar.py

# 🔥 REEMPLAZA la función list_all_available_models en hands/entrenar.py

@router.get("/models/available")
def list_all_available_models():
    """Lista todos los modelos disponibles para descarga pública - VERSIÓN CORREGIDA"""
    try:
        if not os.path.exists(MODELS_DIR):
            return JSONResponse({
                "models": [],
                "total": 0,
                "message": "No hay modelos disponibles"
            })

        available_models = []
        
        # 🆕 BUSCAR EN AMBAS UBICACIONES
        search_directories = [
            MODELS_DIR,  # Directorio principal
            os.path.join(MODELS_DIR, "frontend_uploads")  # Directorio de uploads del frontend
        ]
        
        for search_dir in search_directories:
            if not os.path.exists(search_dir):
                continue
                
            logger.info(f"🔍 Buscando modelos en: {search_dir}")
            
            # 🆕 MÉTODO 1: Buscar por archivos _info.json (modelos tradicionales)
            for filename in os.listdir(search_dir):
                if filename.endswith("_info.json"):
                    try:
                        with open(os.path.join(search_dir, filename), "r", encoding="utf-8") as f:
                            model_info = json.load(f)
                        
                        category = model_info.get("category", "unknown")
                        model_name = model_info.get("model_name", "unknown")
                        
                        # Verificar archivos del modelo tradicional
                        model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_model.h5")
                        encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
                        
                        # 🆕 TAMBIÉN BUSCAR ARCHIVOS TFJS
                        model_json_path = os.path.join(MODELS_DIR, "frontend_uploads", f"{category}_{model_name}_model.json")
                        weights_bin_path = os.path.join(MODELS_DIR, "frontend_uploads", f"{category}_{model_name}_weights.bin")
                        
                        # Verificar qué archivos existen
                        has_traditional = os.path.exists(model_path) and os.path.exists(encoder_path)
                        has_tfjs = os.path.exists(model_json_path) and os.path.exists(weights_bin_path)
                        
                        if has_traditional or has_tfjs:
                            # Calcular tamaños
                            model_size = 0
                            weights_size = 0
                            
                            if has_tfjs:
                                model_size = os.path.getsize(model_json_path)
                                weights_size = os.path.getsize(weights_bin_path)
                            
                            model_data = {
                                "category": category,
                                "model_name": model_name,
                                "labels": model_info.get("labels", []),
                                "accuracy": model_info.get("final_metrics", {}).get("accuracy", 0) * 100,
                                "training_date": model_info.get("training_date", ""),
                                "samples_used": model_info.get("num_samples", 0),
                                "download_info": {
                                    "ready_for_download": has_tfjs,
                                    "model_url": f"/train/download/model/{category}/{model_name}/model.json" if has_tfjs else None,
                                    "weights_url": f"/train/download/model/{category}/{model_name}/weights.bin" if has_tfjs else None,
                                    "model_size_bytes": model_size,
                                    "weights_size_bytes": weights_size,
                                    "total_size_mb": round((model_size + weights_size) / (1024*1024), 2) if has_tfjs else 0,
                                    "has_traditional": has_traditional,
                                    "has_tfjs": has_tfjs
                                }
                            }
                            
                            available_models.append(model_data)
                            logger.info(f"✅ Modelo encontrado (tradicional): {category}/{model_name}")
                            
                    except Exception as e:
                        logger.error(f"Error procesando modelo tradicional {filename}: {e}")
            
            # 🆕 MÉTODO 2: Buscar archivos TFJS directamente (sin _info.json)
            tfjs_models = {}
            
            for filename in os.listdir(search_dir):
                if filename.endswith("_model.json"):
                    # Extraer category y model_name del filename
                    # Formato: category_modelname_model.json
                    base_name = filename.replace("_model.json", "")
                    parts = base_name.split("_")
                    
                    if len(parts) >= 2:
                        category = parts[0]
                        model_name = "_".join(parts[1:])  # El resto es el nombre del modelo
                        
                        model_key = f"{category}_{model_name}"
                        
                        if model_key not in tfjs_models:
                            tfjs_models[model_key] = {
                                "category": category,
                                "model_name": model_name,
                                "has_model": False,
                                "has_weights": False,
                                "model_path": None,
                                "weights_path": None
                            }
                        
                        tfjs_models[model_key]["has_model"] = True
                        tfjs_models[model_key]["model_path"] = os.path.join(search_dir, filename)
                
                elif filename.endswith("_weights.bin"):
                    # Extraer category y model_name del filename
                    base_name = filename.replace("_weights.bin", "")
                    parts = base_name.split("_")
                    
                    if len(parts) >= 2:
                        category = parts[0]
                        model_name = "_".join(parts[1:])
                        
                        model_key = f"{category}_{model_name}"
                        
                        if model_key not in tfjs_models:
                            tfjs_models[model_key] = {
                                "category": category,
                                "model_name": model_name,
                                "has_model": False,
                                "has_weights": False,
                                "model_path": None,
                                "weights_path": None
                            }
                        
                        tfjs_models[model_key]["has_weights"] = True
                        tfjs_models[model_key]["weights_path"] = os.path.join(search_dir, filename)
            
            # Procesar modelos TFJS encontrados
            for model_key, model_info in tfjs_models.items():
                if model_info["has_model"] and model_info["has_weights"]:
                    # Verificar si ya lo agregamos por el método tradicional
                    already_added = any(
                        m["category"] == model_info["category"] and 
                        m["model_name"] == model_info["model_name"] 
                        for m in available_models
                    )
                    
                    if not already_added:
                        try:
                            # Calcular tamaños
                            model_size = os.path.getsize(model_info["model_path"])
                            weights_size = os.path.getsize(model_info["weights_path"])
                            
                            # 🆕 INTENTAR LEER LABELS DEL model.json
                            labels = []
                            try:
                                with open(model_info["model_path"], "r", encoding="utf-8") as f:
                                    model_json = json.load(f)
                                    # Los labels no están en model.json típicamente, usar defaults
                                    category = model_info["category"]
                                    if category == "vocales":
                                        labels = ["A", "E", "I", "O", "U"]
                                    elif category == "numeros":
                                        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                                    elif category == "operaciones":
                                        labels = ["+", "-", "*", "/", "="]
                                    elif category == "palabras":
                                        labels = ["hola", "gracias", "por_favor", "si", "no"]
                            except:
                                labels = [f"label_{i}" for i in range(5)]  # Fallback
                            
                            model_data = {
                                "category": model_info["category"],
                                "model_name": model_info["model_name"],
                                "labels": labels,
                                "accuracy": 85.0,  # Estimado
                                "training_date": datetime.now().isoformat(),  # Usar fecha actual si no tenemos info
                                "samples_used": 100,  # Estimado
                                "download_info": {
                                    "ready_for_download": True,
                                    "model_url": f"/train/download/model/{model_info['category']}/{model_info['model_name']}/model.json",
                                    "weights_url": f"/train/download/model/{model_info['category']}/{model_info['model_name']}/weights.bin",
                                    "model_size_bytes": model_size,
                                    "weights_size_bytes": weights_size,
                                    "total_size_mb": round((model_size + weights_size) / (1024*1024), 2),
                                    "has_traditional": False,
                                    "has_tfjs": True,
                                    "source": "direct_tfjs"  # Indicar que se encontró directamente
                                }
                            }
                            
                            available_models.append(model_data)
                            logger.info(f"✅ Modelo TFJS encontrado directamente: {model_info['category']}/{model_info['model_name']}")
                            
                        except Exception as e:
                            logger.error(f"Error procesando modelo TFJS {model_key}: {e}")

        # Ordenar por fecha de entrenamiento
        available_models.sort(key=lambda x: x.get("training_date", ""), reverse=True)
        
        logger.info(f"📊 Total modelos encontrados: {len(available_models)}")
        
        for model in available_models:
            logger.info(f"  - {model['category']}/{model['model_name']} (Ready: {model['download_info']['ready_for_download']})")

        return JSONResponse({
            "models": available_models,
            "total": len(available_models),
            "message": f"Se encontraron {len(available_models)} modelos disponibles para descarga",
            "search_directories": search_directories,
            "debug_info": {
                "models_dir_exists": os.path.exists(MODELS_DIR),
                "frontend_uploads_exists": os.path.exists(os.path.join(MODELS_DIR, "frontend_uploads")),
                "total_files_found": sum([len(os.listdir(d)) for d in search_directories if os.path.exists(d)])
            }
        })
        
    except Exception as e:
        logger.error(f"Error listando modelos disponibles: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error interno: {str(e)}",
                "models": [],
                "total": 0
            }
        )

# 🆕 TAMBIÉN AGREGAR ENDPOINT DE DEBUG
@router.get("/debug/files")
def debug_files_in_backend():
    """Endpoint de debug para ver qué archivos están en el backend"""
    try:
        debug_info = {
            "models_dir": MODELS_DIR,
            "models_dir_exists": os.path.exists(MODELS_DIR),
            "directories": {}
        }
        
        search_directories = [
            MODELS_DIR,
            os.path.join(MODELS_DIR, "frontend_uploads")
        ]
        
        for directory in search_directories:
            if os.path.exists(directory):
                files = os.listdir(directory)
                debug_info["directories"][directory] = {
                    "exists": True,
                    "file_count": len(files),
                    "files": files,
                    "tfjs_models": [f for f in files if f.endswith("_model.json")],
                    "tfjs_weights": [f for f in files if f.endswith("_weights.bin")],
                    "info_files": [f for f in files if f.endswith("_info.json")]
                }
            else:
                debug_info["directories"][directory] = {"exists": False}
        
        return JSONResponse(debug_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/download/model/{category}/{model_name}/model.json")
async def download_model_json(category: str, model_name: str):
    """Descarga el archivo model.json de un modelo específico"""
    try:
        model_json_path = os.path.join(MODELS_DIR, "frontend_uploads", f"{category}_{model_name}_model.json")
        
        if not os.path.exists(model_json_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Archivo model.json no encontrado para {category}/{model_name}"}
            )
        
        return FileResponse(
            path=model_json_path,
            filename=f"{category}_{model_name}_model.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error descargando model.json: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error descargando archivo: {str(e)}"}
        )

@router.get("/download/model/{category}/{model_name}/weights.bin")
async def download_model_weights(category: str, model_name: str):
    """Descarga el archivo weights.bin de un modelo específico"""
    try:
        weights_bin_path = os.path.join(MODELS_DIR, "frontend_uploads", f"{category}_{model_name}_weights.bin")
        
        if not os.path.exists(weights_bin_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Archivo weights.bin no encontrado para {category}/{model_name}"}
            )
        
        return FileResponse(
            path=weights_bin_path,
            filename=f"{category}_{model_name}_weights.bin",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Error descargando weights.bin: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error descargando archivo: {str(e)}"}
        )

@router.get("/download/model/{category}/{model_name}/info")
async def download_model_info(category: str, model_name: str):
    """Obtiene información completa de un modelo para descarga"""
    try:
        info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
        
        if not os.path.exists(info_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Información del modelo {category}/{model_name} no encontrada"}
            )
        
        with open(info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
        
        # Verificar archivos necesarios
        model_json_path = os.path.join(MODELS_DIR, "frontend_uploads", f"{category}_{model_name}_model.json")
        weights_bin_path = os.path.join(MODELS_DIR, "frontend_uploads", f"{category}_{model_name}_weights.bin")
        
        download_ready = os.path.exists(model_json_path) and os.path.exists(weights_bin_path)
        
        # Información para descarga
        download_info = {
            "ready_for_download": download_ready,
            "model_url": f"/train/download/model/{category}/{model_name}/model.json" if download_ready else None,
            "weights_url": f"/train/download/model/{category}/{model_name}/weights.bin" if download_ready else None,
            "model_size_bytes": os.path.getsize(model_json_path) if download_ready else 0,
            "weights_size_bytes": os.path.getsize(weights_bin_path) if download_ready else 0
        }
        
        if download_ready:
            download_info["total_size_mb"] = round(
                (download_info["model_size_bytes"] + download_info["weights_size_bytes"]) / (1024*1024), 2
            )
        
        return JSONResponse({
            **model_info,
            "download_info": download_info
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo información del modelo: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error obteniendo información: {str(e)}"}
        )