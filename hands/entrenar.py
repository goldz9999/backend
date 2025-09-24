from fastapi import APIRouter

router = APIRouter()

# Endpoint para entrenar el modelo
@router.post("/")
def entrenar_model():
    # Aquí pondrás tu lógica de entrenamiento con IA
    return {"message": "Entrenamiento iniciado ✅"}
