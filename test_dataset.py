#!/usr/bin/env python
"""
Script de prueba para verificar el dataset y el procesamiento de datos.
Ejecuta verificaciones sin necesidad de GPU.
"""

import json
import sys
from pathlib import Path


def test_dataset_structure(data_path: str, verbose: bool = True):
    """
    Verifica que el dataset tenga la estructura correcta.
    
    Estructura esperada:
    - Pregunta: str
    - Razonamiento_Complejo: str
    - Respuesta: str
    - Archivo_fuente: str (opcional)
    """
    print(f"\n{'='*60}")
    print(f"  Verificando dataset: {data_path}")
    print(f"{'='*60}\n")
    
    # Cargar dataset
    try:
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {data_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: JSON inválido - {e}")
        return False
    
    print(f"✓ Dataset cargado: {len(data)} ejemplos")
    
    # Campos requeridos
    required_fields = ['Pregunta', 'Razonamiento_Complejo', 'Respuesta']
    optional_fields = ['Archivo_fuente']
    
    # Verificar estructura
    valid_count = 0
    invalid_examples = []
    
    for i, item in enumerate(data):
        missing = [f for f in required_fields if f not in item or not item[f]]
        if missing:
            invalid_examples.append({
                'index': i,
                'missing': missing,
                'item': item
            })
        else:
            valid_count += 1
    
    print(f"✓ Ejemplos válidos: {valid_count}/{len(data)}")
    
    if invalid_examples:
        print(f"\n⚠️  {len(invalid_examples)} ejemplos con campos faltantes:")
        for inv in invalid_examples[:5]:  # Mostrar máximo 5
            print(f"   - Índice {inv['index']}: falta {inv['missing']}")
        if len(invalid_examples) > 5:
            print(f"   ... y {len(invalid_examples) - 5} más")
    
    # Estadísticas de longitud
    if valid_count > 0:
        valid_items = [d for d in data if all(f in d and d[f] for f in required_fields)]
        
        pregunta_lens = [len(d['Pregunta']) for d in valid_items]
        razonamiento_lens = [len(d['Razonamiento_Complejo']) for d in valid_items]
        respuesta_lens = [len(d['Respuesta']) for d in valid_items]
        
        print(f"\n📊 Estadísticas de longitud (caracteres):")
        print(f"   Pregunta:            min={min(pregunta_lens):,}, max={max(pregunta_lens):,}, avg={sum(pregunta_lens)//len(pregunta_lens):,}")
        print(f"   Razonamiento_Complejo: min={min(razonamiento_lens):,}, max={max(razonamiento_lens):,}, avg={sum(razonamiento_lens)//len(razonamiento_lens):,}")
        print(f"   Respuesta:           min={min(respuesta_lens):,}, max={max(respuesta_lens):,}, avg={sum(respuesta_lens)//len(respuesta_lens):,}")
    
    # Mostrar ejemplo
    if verbose and valid_count > 0:
        example = next(d for d in data if all(f in d and d[f] for f in required_fields))
        print(f"\n📝 Ejemplo del dataset:")
        print(f"{'─'*60}")
        print(f"Pregunta:\n{example['Pregunta'][:300]}...")
        print(f"\nRazonamiento (primeros 300 chars):\n{example['Razonamiento_Complejo'][:300]}...")
        print(f"\nRespuesta (primeros 300 chars):\n{example['Respuesta'][:300]}...")
        if 'Archivo_fuente' in example:
            print(f"\nFuente: {example['Archivo_fuente']}")
        print(f"{'─'*60}")
    
    return valid_count == len(data)


def test_tokenization(data_path: str, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """
    Prueba la tokenización del dataset con un modelo específico.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ Error: transformers no está instalado")
        print("   Ejecuta: pip install transformers")
        return False
    
    print(f"\n{'='*60}")
    print(f"  Probando tokenización con: {model_name}")
    print(f"{'='*60}\n")
    
    # Cargar datos
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Cargar tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer cargado: {model_name}")
    except Exception as e:
        print(f"❌ Error cargando tokenizer: {e}")
        print("   Puedes probar con otro modelo o ejecutar sin esta prueba")
        return False
    
    # Template de chat
    chat_template_llama3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = chat_template_llama3
    
    # Probar con algunos ejemplos
    sample_size = min(10, len(data))
    token_counts = []
    
    for i, item in enumerate(data[:sample_size]):
        if not all(k in item and item[k] for k in ['Pregunta', 'Razonamiento_Complejo', 'Respuesta']):
            continue
            
        response = f"## Thinking\n\n{item['Razonamiento_Complejo']}\n\n## Final Response\n\n{item['Respuesta']}"
        
        messages = [
            {"role": "user", "content": item['Pregunta']},
            {"role": "assistant", "content": response}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))
        
        if i == 0:
            print(f"\n📝 Ejemplo tokenizado:")
            print(f"{'─'*60}")
            print(f"{text[:500]}...")
            print(f"{'─'*60}")
            print(f"   Tokens: {len(tokens)}")
    
    if token_counts:
        print(f"\n📊 Estadísticas de tokens (muestra de {len(token_counts)} ejemplos):")
        print(f"   Min: {min(token_counts):,} tokens")
        print(f"   Max: {max(token_counts):,} tokens")
        print(f"   Promedio: {sum(token_counts)//len(token_counts):,} tokens")
    
    return True


def create_test_sample(input_path: str, output_path: str, n_samples: int = 20):
    """
    Crea un subset pequeño del dataset para pruebas rápidas.
    """
    print(f"\n{'='*60}")
    print(f"  Creando muestra de prueba")
    print(f"{'='*60}\n")
    
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)
    
    # Filtrar solo ejemplos válidos
    valid_data = [
        d for d in data 
        if all(k in d and d[k] for k in ['Pregunta', 'Razonamiento_Complejo', 'Respuesta'])
    ]
    
    sample = valid_data[:n_samples]
    
    # Guardar
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Muestra creada: {output_path}")
    print(f"  {len(sample)} ejemplos guardados")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pruebas del dataset médico')
    parser.add_argument('--data_path', type=str, default='./merged_medical_datasets_v2.json',
                       help='Ruta al dataset JSON')
    parser.add_argument('--test_tokenization', action='store_true',
                       help='Probar tokenización (requiere transformers)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                       help='Modelo para prueba de tokenización')
    parser.add_argument('--create_sample', action='store_true',
                       help='Crear muestra pequeña para pruebas')
    parser.add_argument('--sample_output', type=str, default='./data/test_sample.json',
                       help='Ruta de salida para la muestra')
    parser.add_argument('--sample_size', type=int, default=20,
                       help='Número de ejemplos en la muestra')
    parser.add_argument('--quiet', action='store_true',
                       help='Menos output verboso')
    
    args = parser.parse_args()
    
    # Ejecutar pruebas
    all_passed = True
    
    # 1. Verificar estructura
    if not test_dataset_structure(args.data_path, verbose=not args.quiet):
        all_passed = False
    
    # 2. Probar tokenización (opcional)
    if args.test_tokenization:
        if not test_tokenization(args.data_path, args.model_name):
            all_passed = False
    
    # 3. Crear muestra (opcional)
    if args.create_sample:
        if not create_test_sample(args.data_path, args.sample_output, args.sample_size):
            all_passed = False
    
    # Resumen
    print(f"\n{'='*60}")
    if all_passed:
        print("  ✅ Todas las pruebas pasaron")
    else:
        print("  ⚠️  Algunas pruebas fallaron")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

