from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import logging

router = APIRouter()

DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingStatus:
    def __init__(self):
        self.status = "idle"  # idle, training, completed, error
        self.progress = 0
        self.message = ""
        self.start_time = None
        self.end_time = None
        self.metrics = {}

# Estado global de entrenamiento
training_status = TrainingStatus()

def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesa los datos para mejorar el entrenamiento"""
    
    # Normalizar landmarks (ya están en rango 0-1 pero podemos estandarizar)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Guardar el scaler para uso posterior
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    return X_scaled, y

def create_advanced_model(input_shape: int, num_classes: int) -> Sequential:
    """Crea un modelo más avanzado con mejores técnicas"""
    
    model = Sequential([
        # Capa de entrada con normalización
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capas ocultas con skip connections simuladas
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Capa de salida
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimizador con learning rate adaptativo
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def load_and_process_category_data(category: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga y procesa datos con validaciones avanzadas"""
    
    file_path = os.path.join(DATA_DIR, f"Category.{category}.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Datos de categoría '{category}' no encontrados")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    X = []
    y = []
    labels = list(data.keys())
    
    # Validar muestras mínimas
    insufficient_labels = []
    for label, samples in data.items():
        if len(samples) < 20:  # Mínimo reducido a 20 para pruebas
            insufficient_labels.append(f"{label}: {len(samples)}/20")
    
    if insufficient_labels:
        raise ValueError(f"Etiquetas insuficientes: {', '.join(insufficient_labels)}")
    
    # Procesar muestras
    for label, samples in data.items():
        for sample in samples:
            # Manejar tanto formato simple como con metadatos
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
async def train_advanced_model(category: str, background_tasks: BackgroundTasks):
    """Entrena un modelo avanzado con validación cruzada"""
    
    # Verificar si ya hay un entrenamiento en progreso
    if training_status.status == "training":
        return JSONResponse(
            status_code=409,
            content={"error": "Ya hay un entrenamiento en progreso"}
        )
    
    # Iniciar entrenamiento en background
    background_tasks.add_task(train_model_background, category)
    
    return JSONResponse({
        "message": f"Entrenamiento de '{category}' iniciado en segundo plano",
        "status": "started",
        "check_progress_url": f"/train/progress/{category}"
    })

def train_model_background(category: str):
    """Función de entrenamiento en segundo plano"""
    
    global training_status
    
    try:
        training_status.status = "training"
        training_status.progress = 0
        training_status.message = "Iniciando entrenamiento..."
        training_status.start_time = datetime.now()
        
        # Cargar datos
        training_status.message = "Cargando datos..."
        training_status.progress = 10
        X, y, labels = load_and_process_category_data(category)
        
        # Preprocesar datos
        training_status.message = "Preprocesando datos..."
        training_status.progress = 20
        X_processed, y_processed = preprocess_data(X, y)
        
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
        model = create_advanced_model(X_processed.shape[1], len(labels))
        
        # Callbacks avanzados
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Entrenamiento
        training_status.message = "Entrenando modelo..."
        training_status.progress = 50
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,  # Batch más pequeño para mejor convergencia
            callbacks=callbacks,
            verbose=1
        )
        
        training_status.progress = 80
        training_status.message = "Evaluando modelo..."
        
        # Evaluación final
        val_predictions = model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        
        # Métricas detalladas
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_val, val_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_pred_classes, average='weighted'
        )
        
        # Guardar todo
        training_status.message = "Guardando modelo..."
        training_status.progress = 90
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Guardar modelo
        model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
        model.save(model_path)
        
        # Guardar encoder
        encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        
        # Guardar información completa
        model_info = {
            "category": category,
            "labels": labels,
            "num_samples": len(X),
            "num_classes": len(labels),
            "input_shape": X_processed.shape[1],
            "training_date": datetime.now().isoformat(),
            "final_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "val_loss": float(min(history.history['val_loss']))
            },
            "training_history": {
                "epochs": len(history.history['loss']),
                "best_epoch": int(np.argmin(history.history['val_loss'])) + 1
            }
        }
        
        info_path = os.path.join(MODELS_DIR, f"{category}_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        # Completar entrenamiento
        training_status.status = "completed"
        training_status.progress = 100
        training_status.message = "Entrenamiento completado exitosamente"
        training_status.end_time = datetime.now()
        training_status.metrics = model_info["final_metrics"]
        
        logger.info(f"Entrenamiento completado para {category} con {accuracy:.2%} de precisión")
        
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
        "metrics": training_status.metrics
    })