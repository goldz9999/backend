from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hands.recolectar import router as recolectar_router
from hands.entrenar import router as entrenar_router
from hands.predecir import router as predecir_router
import os
import json

app = FastAPI(
    title="🤖 IA de Reconocimiento de Señas",
    description="Backend para sistema de reconocimiento de lenguaje de señas usando MediaPipe y TensorFlow",
    version="1.0.0"
)

# Configurar CORS para permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar los routers
app.include_router(recolectar_router, prefix="/collect", tags=["📊 Recolección de Datos"])
app.include_router(entrenar_router, prefix="/train", tags=["🧠 Entrenamiento"])
app.include_router(predecir_router, prefix="/predict", tags=["🔮 Predicción"])

@app.get("/")
def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "🚀 Backend de IA para Reconocimiento de Señas",
        "version": "1.0.0",
        "status": "online",
        "features": {
            "data_collection": "✅ Recolección con MediaPipe",
            "training": "✅ Entrenamiento con TensorFlow", 
            "prediction": "✅ Predicción en tiempo real",
            "categories": ["vocales", "numeros", "operaciones", "palabras"]
        },
        "endpoints": {
            "collect": "/collect/sample - POST: Recolectar muestras",
            "status": "/collect/status/{category} - GET: Ver estado de recolección",
            "train": "/train/{category} - POST: Entrenar modelo",
            "predict": "/predict/{category} - POST: Hacer predicción"
        }
    }

@app.get("/health")
def health_check():
    """Endpoint de salud del sistema"""
    try:
        # Verificar directorios importantes
        data_dir_exists = os.path.exists("data")
        models_dir_exists = os.path.exists("models")
        
        # Contar archivos de datos
        data_files = 0
        if data_dir_exists:
            data_files = len([f for f in os.listdir("data") if f.endswith('.json')])
        
        # Contar modelos
        model_files = 0
        if models_dir_exists:
            model_files = len([f for f in os.listdir("models") if f.endswith('_model.h5')])
        
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Esto se actualizaría automáticamente en producción
            "directories": {
                "data": data_dir_exists,
                "models": models_dir_exists
            },
            "files": {
                "datasets": data_files,
                "trained_models": model_files
            },
            "dependencies": {
                "mediapipe": "✅ Instalado",
                "tensorflow": "✅ Instalado",
                "opencv": "✅ Instalado"
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/stats")
def get_system_stats():
    """Obtiene estadísticas generales del sistema"""
    try:
        stats = {
            "categories": {},
            "total_samples": 0,
            "total_models": 0,
            "ready_categories": []
        }
        
        # Contar datos por categoría
        if os.path.exists("data"):
            for filename in os.listdir("data"):
                if filename.startswith("Category.") and filename.endswith(".json"):
                    category = filename.replace("Category.", "").replace(".json", "")
                    
                    with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    category_samples = sum(len(samples) for samples in data.values())
                    stats["categories"][category] = {
                        "labels": len(data),
                        "samples": category_samples,
                        "ready_for_training": all(len(samples) >= 30 for samples in data.values())
                    }
                    stats["total_samples"] += category_samples
        
        # Contar modelos entrenados
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith("_model.h5")]
            stats["total_models"] = len(model_files)
            
            # Categorías listas para predicción
            for model_file in model_files:
                category = model_file.replace("_model.h5", "")
                encoder_exists = os.path.exists(os.path.join("models", f"{category}_encoder.pkl"))
                if encoder_exists:
                    stats["ready_categories"].append(category)
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@app.get("/categories")
def get_available_categories():
    """Lista todas las categorías disponibles"""
    categories = {
        "vocales": {
            "name": "Vocales",
            "description": "Letras A, E, I, O, U en lenguaje de señas",
            "expected_labels": ["A", "E", "I", "O", "U"]
        },
        "numeros": {
            "name": "Números", 
            "description": "Números del 0 al 9 en lenguaje de señas",
            "expected_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        },
        "operaciones": {
            "name": "Operaciones Matemáticas",
            "description": "Símbolos de operaciones matemáticas",
            "expected_labels": ["+", "-", "*", "/", "="]
        },
        "palabras": {
            "name": "Palabras Básicas",
            "description": "Palabras comunes en lenguaje de señas",
            "expected_labels": ["hola", "gracias", "por_favor", "si", "no"]
        }
    }
    
    # Agregar estado actual de cada categoría
    for category_id, category_info in categories.items():
        data_file = os.path.join("data", f"Category.{category_id}.json")
        model_file = os.path.join("models", f"{category_id}_model.h5")
        
        category_info["has_data"] = os.path.exists(data_file)
        category_info["has_model"] = os.path.exists(model_file)
        category_info["samples_count"] = 0
        
        if category_info["has_data"]:
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                category_info["samples_count"] = sum(len(samples) for samples in data.values())
                category_info["labels_with_data"] = list(data.keys())
            except:
                pass
    
    return {
        "categories": categories,
        "total_categories": len(categories)
    }

@app.delete("/reset")
def reset_system():
    """Reinicia el sistema eliminando todos los datos y modelos (usar con cuidado)"""
    try:
        deleted_files = []
        
        # Eliminar archivos de datos
        if os.path.exists("data"):
            for filename in os.listdir("data"):
                if filename.endswith(".json"):
                    file_path = os.path.join("data", filename)
                    os.remove(file_path)
                    deleted_files.append(f"data/{filename}")
        
        # Eliminar modelos
        if os.path.exists("models"):
            for filename in os.listdir("models"):
                file_path = os.path.join("models", filename)
                os.remove(file_path)
                deleted_files.append(f"models/{filename}")
        
        return {
            "message": "🔄 Sistema reiniciado completamente",
            "deleted_files": deleted_files,
            "total_deleted": len(deleted_files),
            "warning": "Todos los datos y modelos han sido eliminados"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reiniciando sistema: {str(e)}")

@app.get("/docs-info")
def get_documentation_info():
    """Información sobre cómo usar el sistema"""
    return {
        "title": "🤖 Guía de Uso - IA de Reconocimiento de Señas",
        "workflow": {
            "step_1": {
                "title": "📊 Recolección de Datos",
                "description": "Recolectar 30 muestras por cada etiqueta usando la cámara",
                "endpoint": "POST /collect/sample",
                "required_files": ["imagen de la seña"],
                "parameters": ["category", "label"]
            },
            "step_2": {
                "title": "🧠 Entrenamiento",
                "description": "Entrenar el modelo de IA con las muestras recolectadas",
                "endpoint": "POST /train/{category}",
                "requirements": "Mínimo 30 muestras por etiqueta"
            },
            "step_3": {
                "title": "🔮 Predicción",
                "description": "Usar el modelo entrenado para reconocer señas",
                "endpoint": "POST /predict/{category}",
                "requirements": "Modelo entrenado disponible"
            }
        },
        "tips": {
            "data_collection": [
                "Usa buena iluminación",
                "Mantén la mano centrada en la cámara",
                "Varía ligeramente la posición entre muestras",
                "Asegúrate de que la seña sea clara y consistente"
            ],
            "training": [
                "Espera a tener todas las muestras antes de entrenar",
                "El entrenamiento puede tomar algunos minutos",
                "Revisa la precisión del modelo antes de usarlo"
            ],
            "prediction": [
                "Usa las mismas condiciones de iluminación que en el entrenamiento",
                "Mantén la mano estable durante la predicción",
                "Si la confianza es baja, recolecta más muestras"
            ]
        },
        "categories_suggested": {
            "vocales": "Ideal para comenzar - 5 etiquetas simples",
            "numeros": "Números 0-9 - útil para matemáticas básicas",
            "operaciones": "Símbolos matemáticos +, -, *, /, =",
            "palabras": "Palabras comunes como hola, gracias, etc."
        }
    }

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "detail": str(exc),
            "message": "Ha ocurrido un error inesperado. Revisa los logs para más detalles."
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor de IA de Reconocimiento de Señas...")
    print("📖 Documentación disponible en: http://127.0.0.1:8000/docs")
    print("📊 Estadísticas del sistema en: http://127.0.0.1:8000/stats")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )