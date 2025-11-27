#!/usr/bin/env python3
"""
Script para dividir el dataset en train/val/test y preparar para entrenamiento.
Optimizado para Google Colab Pro+ con GPU T4/V100.
"""

import json
import random
from pathlib import Path

def split_dataset(
    input_file: str = "merged_medical_datasets_v2.json",
    train_samples: int = 800,
    val_samples: int = 100,
    test_samples: int = 100,
    output_dir: str = "./data",
    seed: int = 42
):
    """
    Divide el dataset en train/val/test.
    
    Para Colab Pro+ con GPU T4 (16GB VRAM):
    - Train: 800 samples (suficiente para ver convergencia)
    - Val: 100 samples (10% del train set)
    - Test: 100 samples (para evaluación final)
    """
    print("="*70)
    print("PREPARACIÓN DE DATASET PARA COLAB PRO+")
    print("="*70)
    
    # Configurar seed
    random.seed(seed)
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Cargar dataset completo
    print(f"\n📂 Cargando dataset desde: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"✓ Dataset cargado: {len(full_data)} ejemplos totales")
    
    # Validar que hay suficientes datos
    total_needed = train_samples + val_samples + test_samples
    if len(full_data) < total_needed:
        print(f"⚠️  Warning: Solo hay {len(full_data)} ejemplos, necesitas {total_needed}")
        print(f"   Ajustando proporcionalmente...")
        
        # Ajustar proporcionalmente
        ratio = len(full_data) / total_needed
        train_samples = int(train_samples * ratio)
        val_samples = int(val_samples * ratio)
        test_samples = int(test_samples * ratio)
        
        print(f"   Nuevo: train={train_samples}, val={val_samples}, test={test_samples}")
    
    # Mezclar datos
    print("\n🔀 Mezclando datos...")
    shuffled_data = full_data.copy()
    random.shuffle(shuffled_data)
    
    # Dividir
    train_data = shuffled_data[:train_samples]
    val_data = shuffled_data[train_samples:train_samples + val_samples]
    test_data = shuffled_data[train_samples + val_samples:train_samples + val_samples + test_samples]
    
    # Guardar splits
    print("\n💾 Guardando splits...")
    
    train_path = Path(output_dir) / "train.json"
    val_path = Path(output_dir) / "val.json"
    test_path = Path(output_dir) / "test.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Train set guardado: {train_path} ({len(train_data)} ejemplos)")
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Val set guardado: {val_path} ({len(val_data)} ejemplos)")
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Test set guardado: {test_path} ({len(test_data)} ejemplos)")
    
    # Estadísticas
    print("\n" + "="*70)
    print("RESUMEN DE SPLITS")
    print("="*70)
    print(f"Train:      {len(train_data):4d} ejemplos ({len(train_data)/len(full_data)*100:.1f}%)")
    print(f"Validation: {len(val_data):4d} ejemplos ({len(val_data)/len(full_data)*100:.1f}%)")
    print(f"Test:       {len(test_data):4d} ejemplos ({len(test_data)/len(full_data)*100:.1f}%)")
    print(f"Total:      {len(train_data)+len(val_data)+len(test_data):4d} ejemplos")
    print("="*70)
    
    # Calcular estadísticas de longitud
    def get_lengths(data):
        lengths = []
        for item in data:
            q = item.get('Pregunta', '')
            r = item.get('Razonamiento_Complejo', '')
            a = item.get('Respuesta', '')
            total = len(q) + len(r) + len(a)
            lengths.append(total)
        return lengths
    
    train_lengths = get_lengths(train_data)
    
    print(f"\n📊 Estadísticas de longitud (caracteres):")
    print(f"   Promedio: {sum(train_lengths)/len(train_lengths):.0f} chars")
    print(f"   Mínimo:   {min(train_lengths)} chars")
    print(f"   Máximo:   {max(train_lengths)} chars")
    print(f"   Percentil 95: {sorted(train_lengths)[int(len(train_lengths)*0.95)]} chars")
    
    # Estimación de tokens (aproximadamente 4 chars = 1 token en español)
    avg_tokens = sum(train_lengths) / len(train_lengths) / 4
    print(f"   Tokens estimados por ejemplo: ~{avg_tokens:.0f} tokens")
    
    print("\n✅ ¡Dataset preparado exitosamente!")
    
    return {
        'train_path': str(train_path),
        'val_path': str(val_path),
        'test_path': str(test_path),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dividir dataset para entrenamiento')
    parser.add_argument('--input', type=str, default='merged_medical_datasets_v2.json',
                        help='Archivo de entrada')
    parser.add_argument('--train_samples', type=int, default=800,
                        help='Número de ejemplos de entrenamiento')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='Número de ejemplos de validación')
    parser.add_argument('--test_samples', type=int, default=100,
                        help='Número de ejemplos de test')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directorio de salida')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para reproducibilidad')
    
    args = parser.parse_args()
    
    split_dataset(
        input_file=args.input,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )