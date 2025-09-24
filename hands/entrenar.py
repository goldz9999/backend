from fastapi import APIRouter
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import pickle

router = APIRouter()

DATA_DIR = "data"
MODELS_DIR = "models"
SAMPLES_REQUIRED = 30

def load_category_data(category: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga los datos de una categoría específica"""
    
    file_path = os.path.join(DATA_DIR, f"Category.{category}.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontraron datos para la categoría '{category}'")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    X = []  # Features (landmarks)
    y = []  # Labels
    labels = list(data.keys())
    
    # Verificar que todas las etiquetas tengan suficientes muestras
    insufficient_labels = []
    for label, samples in data.items():
        if len(samples) < SAMPLES_REQUIRED:
            insufficient_labels.append(f"{label}: {len(samples)}/{SAMPLES_REQUIRED}")
    
    if insufficient_labels:
        raise ValueError(f"Etiquetas con muestras insuficientes: {', '.join(insufficient_labels)}")
    
    # Convertir datos a arrays numpy
    for label, samples in data.items():
        for sample in samples:
            if len(sample) == 126:  # 2 manos * 21 landmarks * 3 coordenadas
                X.append(sample)
                y.append(label)
    
    return np.array(X), np.array(y), labels

def create_model(input_shape: int, num_classes: int) -> Sequential:
    """Crea el modelo de red neuronal"""
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@router.post("/{category}")
def train_category_model(category: str):
    """Entrena el modelo para una categoría específica"""
    
    try:
        # Cargar datos
        X, y, labels = load_category_data(category)
        
        if len(X) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": f"No hay datos suficientes para entrenar la categoría '{category}'"}
            )
        
        # Codificar labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Dividir datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Crear modelo
        model = create_model(X.shape[1], len(labels))
        
        # Callbacks para entrenamiento
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Entrenar modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Crear directorio de modelos si no existe
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Guardar modelo
        model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
        model.save(model_path)
        
        # Guardar label encoder
        encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        
        # Guardar información del modelo
        model_info = {
            "category": category,
            "labels": labels,
            "num_samples": len(X),
            "num_classes": len(labels),
            "input_shape": X.shape[1],
            "final_accuracy": float(history.history['val_accuracy'][-1]),
            "final_loss": float(history.history['val_loss'][-1])
        }
        
        info_path = os.path.join(MODELS_DIR, f"{category}_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)
        
        return JSONResponse({
            "message": f"Modelo para '{category}' entrenado exitosamente ✅",
            "category": category,
            "labels": labels,
            "samples_used": len(X),
            "final_accuracy": round(model_info["final_accuracy"] * 100, 2),
            "model_path": model_path,
            "ready_for_prediction": True
        })
        
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error durante el entrenamiento: {str(e)}"}
        )

@router.get("/status")
def get_training_status():
    """Obtiene el estado de todos los modelos entrenados"""
    
    if not os.path.exists(MODELS_DIR):
        return JSONResponse({
            "models": {},
            "total_models": 0
        })
    
    models_info = {}
    
    # Buscar archivos de información de modelos
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith("_info.json"):
            category = filename.replace("_info.json", "")
            
            with open(os.path.join(MODELS_DIR, filename), "r", encoding="utf-8") as f:
                model_info = json.load(f)
            
            # Verificar si el archivo del modelo existe
            model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
            encoder_path = os.path.join(MODELS_DIR, f"{category}_encoder.pkl")
            
            models_info[category] = {
                **model_info,
                "model_exists": os.path.exists(model_path),
                "encoder_exists": os.path.exists(encoder_path),
                "ready_for_prediction": os.path.exists(model_path) and os.path.exists(encoder_path)
            }
    
    return JSONResponse({
        "models": models_info,
        "total_models": len(models_info)
    })

@router.delete("/{category}")
def delete_model(category: str):
    """Elimina un modelo entrenado"""
    
    files_to_delete = [
        f"{category}_model.h5",
        f"{category}_encoder.pkl",
        f"{category}_info.json"
    ]
    
    deleted_files = []
    
    for filename in files_to_delete:
        file_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_files.append(filename)
    
    if deleted_files:
        return JSONResponse({
            "message": f"Modelo '{category}' eliminado exitosamente",
            "deleted_files": deleted_files
        })
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"No se encontró el modelo para la categoría '{category}'"}
        )