import requests
import json
import random
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def generate_fake_landmarks():
    """Genera landmarks falsos para pruebas (126 valores)"""
    # 2 manos × 21 landmarks × 3 coordenadas = 126 valores
    landmarks = []
    
    # Primera mano (valores entre 0 y 1)
    for _ in range(21):
        x = random.uniform(0.2, 0.8)  # x normalizado
        y = random.uniform(0.2, 0.8)  # y normalizado  
        z = random.uniform(-0.1, 0.1)  # z (profundidad)
        landmarks.extend([x, y, z])
    
    # Segunda mano o ceros si solo hay una
    if random.choice([True, False]):  # 50% probabilidad de segunda mano
        for _ in range(21):
            x = random.uniform(0.2, 0.8)
            y = random.uniform(0.2, 0.8)
            z = random.uniform(-0.1, 0.1)
            landmarks.extend([x, y, z])
    else:
        landmarks.extend([0.0] * 63)  # Rellenar con ceros
    
    return landmarks

def test_collect_sample():
    """Prueba recolección de una muestra"""
    print("\n🖐️ Probando recolección de muestra...")
    
    landmarks = generate_fake_landmarks()
    
    data = {
        "category": "vocales",
        "label": "A",
        "landmarks": landmarks
    }
    
    response = requests.post(f"{BASE_URL}/collect/sample", json=data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_collect_batch():
    """Prueba recolección en lote"""
    print("\n📊 Probando recolección en lote...")
    
    samples = []
    labels = ["A", "E", "I"]
    
    for label in labels:
        for _ in range(3):  # 3 muestras por letra
            samples.append({
                "label": label,
                "landmarks": generate_fake_landmarks()
            })
    
    data = {
        "category": "vocales",
        "samples": samples
    }
    
    response = requests.post(f"{BASE_URL}/collect/batch", json=data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_collection_status():
    """Prueba estado de recolección"""
    print("\n📈 Verificando estado de recolección...")
    
    response = requests.get(f"{BASE_URL}/collect/status/vocales")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def collect_full_dataset():
    """Recolecta un dataset completo para vocales"""
    print("\n🎯 Recolectando dataset completo para vocales...")
    
    labels = ["A", "E", "I", "O", "U"]
    samples_per_label = 30
    
    for label in labels:
        print(f"\n📝 Recolectando 30 muestras para '{label}'...")
        
        batch_samples = []
        for i in range(samples_per_label):
            batch_samples.append({
                "label": label,
                "landmarks": generate_fake_landmarks()
            })
        
        # Enviar en lotes de 10
        for i in range(0, len(batch_samples), 10):
            batch = batch_samples[i:i+10]
            
            data = {
                "category": "vocales",
                "samples": batch
            }
            
            response = requests.post(f"{BASE_URL}/collect/batch", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Lote enviado - {result['successful_samples']}/10 exitosas")
            else:
                print(f"   ❌ Error en lote: {response.status_code}")

def test_train_model():
    """Prueba entrenamiento del modelo"""
    print("\n🧠 Probando entrenamiento del modelo...")
    
    response = requests.post(f"{BASE_URL}/train/vocales")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_prediction():
    """Prueba predicción"""
    print("\n🔮 Probando predicción...")
    
    landmarks = generate_fake_landmarks()
    
    response = requests.post(f"{BASE_URL}/predict/vocales", json=landmarks)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_available_models():
    """Prueba listado de modelos disponibles"""
    print("\n📋 Verificando modelos disponibles...")
    
    response = requests.get(f"{BASE_URL}/predict/available")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def test_system_stats():
    """Prueba estadísticas del sistema"""
    print("\n📊 Verificando estadísticas del sistema...")
    
    response = requests.get(f"{BASE_URL}/stats")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

def run_complete_test():
    """Ejecuta prueba completa del sistema"""
    print("🚀 EJECUTANDO PRUEBAS COMPLETAS DEL SISTEMA")
    print("=" * 60)
    
    tests = [
        ("🖐️ Recolección individual", test_collect_sample),
        ("📊 Recolección en lote", test_collect_batch),
        ("📈 Estado de recolección", test_collection_status),
        ("📊 Estadísticas del sistema", test_system_stats),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✅ PASS" if result else "❌ FAIL"))
        except Exception as e:
            results.append((test_name, f"💥 ERROR: {str(e)}"))
    
    print(f"\n{'='*60}")
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    # Preguntar si quiere dataset completo
    print(f"\n{'='*60}")
    resp = input("¿Quieres recolectar dataset completo y entrenar modelo? (s/N): ")
    
    if resp.lower() == 's':
        collect_full_dataset()
        
        print("\n📊 Verificando estado final...")
        test_collection_status()
        
        print("\n🧠 Iniciando entrenamiento...")
        if test_train_model():
            print("\n🔮 Probando predicción con modelo entrenado...")
            test_prediction()
            test_available_models()

def interactive_menu():
    """Menú interactivo para pruebas"""
    while True:
        print(f"\n{'='*50}")
        print("🧪 MENÚ DE PRUEBAS - BACKEND OPTIMIZADO")
        print("=" * 50)
        print("1. 🖐️  Probar recolección individual")
        print("2. 📊  Probar recolección en lote")
        print("3. 📈  Ver estado de recolección")
        print("4. 🎯  Recolectar dataset completo (30 muestras × 5 vocales)")
        print("5. 🧠  Entrenar modelo")
        print("6. 🔮  Probar predicción")
        print("7. 📋  Ver modelos disponibles")
        print("8. 📊  Ver estadísticas del sistema")
        print("9. 🚀  Ejecutar prueba completa")
        print("0. 👋  Salir")
        
        try:
            choice = input("\n🔢 Selecciona una opción (0-9): ").strip()
            
            if choice == "0":
                print("👋 ¡Hasta luego!")
                break
            elif choice == "1":
                test_collect_sample()
            elif choice == "2":
                test_collect_batch()
            elif choice == "3":
                test_collection_status()
            elif choice == "4":
                collect_full_dataset()
            elif choice == "5":
                test_train_model()
            elif choice == "6":
                test_prediction()
            elif choice == "7":
                test_available_models()
            elif choice == "8":
                test_system_stats()
            elif choice == "9":
                run_complete_test()
            else:
                print("❌ Opción inválida")
                
        except KeyboardInterrupt:
            print("\n👋 Interrumpido por usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")

if __name__ == "__main__":
    print("🤖 SISTEMA DE PRUEBAS - BACKEND OPTIMIZADO SIN MEDIAPIPE")
    print("Nota: Este backend recibe landmarks ya procesados desde el frontend")
    print("=" * 70)
    
    # Verificar que el servidor esté funcionando
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✅ Servidor detectado y funcionando")
            interactive_menu()
        else:
            print(f"❌ Servidor responde pero con error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servidor")
        print("   Asegúrate de que esté ejecutando: python main.py")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")