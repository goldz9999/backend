from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime

router = APIRouter()

DATA_DIR = "data"
SAMPLES_PER_LABEL = 30

class LandmarkSample(BaseModel):
    category: str
    label: str
    landmarks: List[float]  # Array de 126 valores (2 manos × 21 landmarks × 3 coords)
    timestamp: str = None
    metadata: Dict[str, Any] = {}

@router.post("/sample/landmarks")
async def collect_landmarks_sample(sample: LandmarkSample):
    """Recolecta una muestra desde landmarks ya procesados"""
    
    # Validar landmarks
    if len(sample.landmarks) != 126:
        return JSONResponse(
            status_code=400,
            content={"error": f"Se esperan 126 landmarks, recibidos: {len(sample.landmarks)}"}
        )
    
    # Agregar timestamp si no existe
    if not sample.timestamp:
        sample.timestamp = datetime.now().isoformat()
    
    # Crear directorio si no existe
    os.makedirs(DATA_DIR, exist_ok=True)
    
    file_path = os.path.join(DATA_DIR, f"Category.{sample.category}.json")
    
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
    if sample.label not in data:
        data[sample.label] = []
    
    # Verificar límite
    if len(data[sample.label]) >= SAMPLES_PER_LABEL:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Límite de {SAMPLES_PER_LABEL} muestras alcanzado",
                "current_samples": len(data[sample.label])
            }
        )
    
    # Crear entrada completa con metadatos
    entry = {
        "landmarks": sample.landmarks,
        "timestamp": sample.timestamp,
        "metadata": sample.metadata
    }
    
    # Agregar muestra
    data[sample.label].append(entry)
    
    # Guardar datos
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Calcular estadísticas
    total_samples = sum(len(samples) for samples in data.values())
    ready_to_train = all(len(samples) >= SAMPLES_PER_LABEL for samples in data.values())
    
    return JSONResponse({
        "message": f"Muestra {len(data[sample.label])}/{SAMPLES_PER_LABEL} guardada",
        "current_samples": len(data[sample.label]),
        "total_samples": total_samples,
        "labels_count": len(data),
        "ready_to_train": ready_to_train,
        "success": True
    })

@router.post("/batch/landmarks")
async def collect_batch_landmarks(samples: List[LandmarkSample]):
    """Recolecta múltiples muestras de landmarks"""
    
    if len(samples) > 50:  # Limitar lote
        return JSONResponse(
            status_code=400,
            content={"error": "Máximo 50 muestras por lote"}
        )
    
    results = []
    successful = 0
    
    for i, sample in enumerate(samples):
        try:
            # Usar la función individual para cada muestra
            result = await collect_landmarks_sample(sample)
            
            if result.status_code == 200:
                successful += 1
                results.append({"index": i, "success": True, "message": "Guardada correctamente"})
            else:
                results.append({"index": i, "success": False, "error": result.body.decode()})
                
        except Exception as e:
            results.append({"index": i, "success": False, "error": str(e)})
    
    return JSONResponse({
        "total_samples": len(samples),
        "successful": successful,
        "failed": len(samples) - successful,
        "results": results
    })

@router.get("/dataset/{category}/summary")
def get_dataset_summary(category: str):
    """Obtiene resumen completo del dataset"""
    
    file_path = os.path.join(DATA_DIR, f"Category.{category}.json")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "category": category,
            "exists": False,
            "labels": {},
            "summary": {
                "total_samples": 0,
                "total_labels": 0,
                "ready_to_train": False,
                "completion_percentage": 0
            }
        })
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Calcular estadísticas por label
    labels_stats = {}
    total_samples = 0
    
    for label, samples in data.items():
        sample_count = len(samples)
        total_samples += sample_count
        
        last_sample = samples[-1] if samples else None
        last_timestamp = last_sample.get("timestamp") if isinstance(last_sample, dict) else None
        
        labels_stats[label] = {
            "samples": sample_count,
            "progress_percentage": min(100, (sample_count / SAMPLES_PER_LABEL) * 100),
            "ready": sample_count >= SAMPLES_PER_LABEL,
            "last_sample": last_timestamp
        }
    
    # Estadísticas generales
    all_ready = all(stats["ready"] for stats in labels_stats.values()) if labels_stats else False
    total_labels = len(labels_stats)
    completion = (total_samples / (total_labels * SAMPLES_PER_LABEL)) * 100 if total_labels > 0 else 0
    
    return JSONResponse({
        "category": category,
        "exists": True,
        "labels": labels_stats,
        "summary": {
            "total_samples": total_samples,
            "total_labels": total_labels,
            "ready_to_train": all_ready,
            "completion_percentage": min(100, completion)
        }
    })

@router.delete("/clear/{category}")
def clear_category_data(category: str, label: str = None):
    """Limpia datos de una categoría o label específico"""
    
    file_path = os.path.join(DATA_DIR, f"Category.{category}.json")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "message": f"No hay datos para la categoría '{category}'"
        })
    
    if label:
        # Limpiar solo un label específico
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        
        if label in data:
            del data[label]
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            return JSONResponse({
                "message": f"Datos del label '{label}' eliminados de la categoría '{category}'"
            })
        else:
            return JSONResponse({
                "message": f"El label '{label}' no existe en la categoría '{category}'"
            })
    else:
        # Limpiar toda la categoría
        os.remove(file_path)
        return JSONResponse({
            "message": f"Todos los datos de la categoría '{category}' han sido eliminados"
        })


