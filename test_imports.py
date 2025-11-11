#!/usr/bin/env python3
"""
Teste de imports para o Grad-CAM
"""

imports_to_test = [
    'torch',
    'numpy', 
    'cv2',
    'os',
    'matplotlib.pyplot'
]

print("Testando imports...")
print("=" * 50)

for import_name in imports_to_test:
    try:
        if import_name == 'cv2':
            import cv2
            print(f"✓ opencv-python (cv2) - versão {cv2.__version__}")
        elif import_name == 'matplotlib.pyplot':
            import matplotlib.pyplot as plt
            print(f"✓ matplotlib - versão {plt.__version__}")
        else:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'versão não disponível')
            print(f"✓ {import_name} - versão {version}")
    except ImportError as e:
        print(f"✗ {import_name} - FALHA: {e}")
    except Exception as e:
        print(f"✗ {import_name} - ERRO: {e}")

print("=" * 50)
print("Teste concluído!")