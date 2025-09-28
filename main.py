# main.py - CONFIGURACIÓN CORS CORREGIDA PARA RENDER + VERCEL
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

# 🔥 CONFIGURACIÓN CORS CORREGIDA PARA PRODUCCIÓN
origins = [
    # Frontend en Vercel (HTTPS)
    "https://proyecto-hands.vercel.app",
    "https://*.vercel.app",  # Para subdominios de Vercel
    
    # Para desarrollo local
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    
    # Backend en Render (por si hay redirecciones internas)
    "https://backend-c2aj.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # 🚨 CAMBIO: URLs específicas, no "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 🚨 AGREGADO: OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],          # 🚨 AGREGADO: Para exponer headers de respuesta
)

# 🆕 ENDPOINT DE VERIFICACIÓN DE SALUD
@app.get("/")
def root():
    return {
        "message": "Backend de IA funcionando 🚀",
        "status": "online",
        "version": "1.0.0",
        "cors_enabled": True,
        "allowed_origins": origins
    }

# 🆕 ENDPOINT PARA VERIFICAR CONECTIVIDAD
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend_url": "https://backend-c2aj.onrender.com",
        "timestamp": "2024-12-07",
        "services": {
            "collect": "online",
            "train": "online", 
            "predict": "online"
        }
    }

# 🆕 ENDPOINT PARA TESTING CORS
@app.get("/cors-test")
def cors_test():
    return {
        "cors_working": True,
        "message": "Si ves este mensaje, CORS está funcionando correctamente",
        "frontend_allowed": "https://proyecto-hands.vercel.app"
    }

# Registrar routers
app.include_router(recolectar_router, prefix="/collect", tags=["Recolectar"])
app.include_router(entrenar_router, prefix="/train", tags=["Entrenar"])
app.include_router(predecir_router, prefix="/predict", tags=["Predecir"])

# 🚨 MANEJO MEJORADO DE ERRORES CORS
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🚨 Error global: {exc}")
    print(f"🔍 Request URL: {request.url}")
    print(f"🔍 Request method: {request.method}")
    print(f"🔍 Request headers: {dict(request.headers)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "message": f"Error interno: {exc}",
            "request_url": str(request.url),
            "method": request.method
        },
        headers={
            "Access-Control-Allow-Origin": "https://proyecto-hands.vercel.app",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# 🆕 MIDDLEWARE ADICIONAL PARA DEBUG CORS
@app.middleware("http")
async def cors_debug_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    print(f"🌐 Petición desde origen: {origin}")
    print(f"🔗 URL solicitada: {request.url}")
    print(f"📋 Método: {request.method}")
    
    response = await call_next(request)
    
    # Agregar headers CORS manualmente como backup
    if origin == "https://proyecto-hands.vercel.app":
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)