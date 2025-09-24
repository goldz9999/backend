from fastapi import FastAPI
from hands.recolectar import router as recolectar_router
from hands.entrenar import router as entrenar_router
from hands.predecir import router as predecir_router

app = FastAPI(title="IA de Manos")

# Registrar los routers
app.include_router(recolectar_router, prefix="/collect", tags=["Recolectar"])
app.include_router(entrenar_router, prefix="/train", tags=["Entrenar"])
app.include_router(predecir_router, prefix="/predict", tags=["Predecir"])

@app.get("/")
def root():
    return {"message": "Backend de IA funcionando 🚀"}
