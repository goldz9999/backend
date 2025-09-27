# hands/entrenar.py
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
from typing import List, Tuple
import logging

router = APIRouter()

DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Request para frontend -----
class TrainingRequest(BaseModel):
    model_name: str
    epochs: int

# ----- Estado global de entrenamiento -----
class TrainingStatus:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.message = ""
        self.start_time = None
        self.end_time = None
        self.metrics = {}

training_status = TrainingStatus()

# ----- Preprocesamiento -----
def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesa los datos para mejorar el entrenamiento"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Guardar el scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled, y

# ----- Modelo avanzado -----
def create_advanced_model(input_shape: int, num_classes: int) -> Sequential:
    """Crea un modelo más avanzado"""
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

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----- Cargar datos -----
def load_and_process_category_data(category: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

# ----- Entrenamiento -----
@router.post("/{category}/advanced")
async def train_advanced_model(category: str, request: TrainingRequest, background_tasks: BackgroundTasks):
    if training_status.status == "training":
        return JSONResponse(
            status_code=409,
            content={"error": "Ya hay un entrenamiento en progreso"}
        )

    background_tasks.add_task(train_model_background, category, request.model_name, request.epochs)

    return JSONResponse({
        "message": f"Entrenamiento de '{category}' iniciado en segundo plano",
        "model_name": request.model_name,
        "epochs": request.epochs,
        "status": "started",
        "check_progress_url": f"/train/progress/{category}"
    })

def train_model_background(category: str, model_name: str, epochs: int):
    global training_status
    try:
        training_status.status = "training"
        training_status.progress = 0
        training_status.message = f"Iniciando entrenamiento '{model_name}'..."
        training_status.start_time = datetime.now()

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        # Datos
        training_status.message = "Cargando datos..."
        training_status.progress = 10
        X, y, labels = load_and_process_category_data(category)

        # Preprocesar
        training_status.message = "Preprocesando datos..."
        training_status.progress = 20
        X_processed, y_processed = preprocess_data(X, y)

        # Codificar etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_processed)

        # División
        training_status.message = "Dividiendo datos..."
        training_status.progress = 30
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        # Modelo
        training_status.message = "Creando modelo..."
        training_status.progress = 40
        model = create_advanced_model(X_processed.shape[1], len(labels))

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001)
        ]

        # Entrenamiento
        training_status.message = "Entrenando modelo..."
        training_status.progress = 50
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
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

        # Guardar modelo
        training_status.message = "Guardando modelo..."
        training_status.progress = 90
        model_path = os.path.join(MODELS_DIR, f"{category}_{model_name}.h5")
        model.save(model_path)

        # Guardar encoder
        encoder_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)

        # Guardar info
        model_info = {
            "category": category,
            "model_name": model_name,
            "labels": labels,
            "num_samples": len(X),
            "epochs": epochs,
            "training_date": datetime.now().isoformat(),
            "final_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
        }
        info_path = os.path.join(MODELS_DIR, f"{category}_{model_name}_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

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

# ----- Progreso -----
@router.get("/progress/{category}")
def get_training_progress(category: str):
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

# ----- Reset -----
@router.post("/{category}/reset")
async def reset_model(category: str):
    try:
        global training_status

        model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
        encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
        info_path = os.path.join(MODELS_DIR, f"{category}_info.json")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

        for path in [model_path, encoder_path, info_path, scaler_path]:
            if os.path.exists(path):
                os.remove(path)

        training_status = TrainingStatus()

        return JSONResponse({
            "message": f"Modelo y datos de '{category}' reseteados exitosamente",
            "status": "reset"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
