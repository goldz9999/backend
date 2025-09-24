# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Importar routers
from hands.recolectar import router as recolectar_router
from hands.entrenar import router as entrenar_router
from hands.predecir import router as predecir_router

# Crear app FastAPI
app = FastAPI(
    title="IA de Manos",
    description="API para recolectar, entrenar y predecir gestos de manos usando IA",
    version="1.0.0"
)

# Configurar CORS (para uso con frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por tu frontend en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(recolectar_router, prefix="/collect", tags=["Recolectar"])
app.include_router(entrenar_router, prefix="/train", tags=["Entrenar"])
app.include_router(predecir_router, prefix="/predict", tags=["Predecir"])

# Ruta raíz
@app.get("/")
def root():
    return {"message": "Backend de IA funcionando 🚀"}

# Manejo global de errores
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"Error interno: {exc}"}
    )

# Ejecutar con: uvicorn main:app --reload
