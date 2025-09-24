import requests
import json

URL = "http://127.0.0.1:8000"

def test_recolectar():
    # Aquí pruebas enviar una muestra
    payload = {"category": "vocales", "label": "a"}
    r = requests.post(f"{URL}/collect/sample", params=payload)

    print("✅ Respuesta del servidor:")
    print(r.json())  # Muestra lo que devolvió el backend

    # Guardar la respuesta en un archivo local (opcional)
    with open("respuesta.json", "w", encoding="utf-8") as f:
        json.dump(r.json(), f, indent=4, ensure_ascii=False)
    print("📂 Respuesta guardada en respuesta.json")

if __name__ == "__main__":
    test_recolectar()
