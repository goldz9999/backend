from fastapi import APIRouter

router = APIRouter()

# Endpoint para hacer predicciones
@router.post("/")
def predecir(data: dict):
    # Aquí usarías tu modelo entrenado para predecir
    return {"prediction": "ejemplo_prediccion"}
