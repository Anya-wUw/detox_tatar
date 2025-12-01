#!/usr/bin/env python3
"""
Параллельный запуск инференса на нескольких GPU
"""
import subprocess
import sys
import os
import pandas as pd
from pathlib import Path

def split_dataframe(df, num_splits):
    """Разделяет DataFrame на части"""
    chunk_size = len(df) // num_splits
    chunks = []
    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_splits - 1 else len(df)
        chunks.append(df.iloc[start:end].copy())
    return chunks

def main():
    if len(sys.argv) < 3:
        print("Использование: python3 run_inference_parallel.py <checkpoint> <input_file> [gpu1] [gpu2]")
        print("Пример: python3 run_inference_parallel.py checkpoint-1594 test_inputs.tsv 0 1")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    input_file = sys.argv[2]
    
    # Определяем GPU
    if len(sys.argv) >= 5:
        gpu1 = sys.argv[3]
        gpu2 = sys.argv[4]
    else:
        # Автоматически находим свободные GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 3:
                gpu_id = parts[0]
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                mem_free_pct = (1 - mem_used / mem_total) * 100
                if mem_free_pct > 50:  # Свободно больше 50%
                    gpus.append((gpu_id, mem_free_pct))
        
        gpus.sort(key=lambda x: x[1], reverse=True)  # Сортируем по свободной памяти
        if len(gpus) < 2:
            print(f"❌ Недостаточно свободных GPU. Найдено: {len(gpus)}")
            sys.exit(1)
        
        gpu1 = gpus[0][0]
        gpu2 = gpus[1][0]
        print(f"✅ Используем GPU {gpu1} и {gpu2}")
    
    # Загружаем данные
    print(f"Загрузка {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Всего примеров: {len(df)}")
    
    # Разделяем на две части
    df1, df2 = split_dataframe(df, 2)
    print(f"GPU {gpu1}: {len(df1)} примеров")
    print(f"GPU {gpu2}: {len(df2)} примеров")
    
    # Сохраняем временные файлы
    temp_file1 = f"{input_file}.part1.tsv"
    temp_file2 = f"{input_file}.part2.tsv"
    df1.to_csv(temp_file1, sep='\t', index=False, encoding='utf-8')
    df2.to_csv(temp_file2, sep='\t', index=False, encoding='utf-8')
    
    # Определяем выходные файлы (используем временные имена, чтобы не конфликтовать)
    output_file1 = f"outputs_part1_gpu{gpu1}.tsv"
    output_file2 = f"outputs_part2_gpu{gpu2}.tsv"
    
    if 'test' in input_file.lower():
        final_output = "test_outputs.tsv"
    else:
        final_output = "outputs.tsv"
    
    # Создаем скрипты для запуска с правильными выходными файлами
    script1 = f"""
import os
os.environ['INFERENCE_GPU'] = '{gpu1}'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
sys.argv = ['inference_gemma_sft.py', '{checkpoint}', '{temp_file1}']
exec(open('inference_gemma_sft.py').read())
"""
    
    script2 = f"""
import os
os.environ['INFERENCE_GPU'] = '{gpu2}'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
sys.argv = ['inference_gemma_sft.py', '{checkpoint}', '{temp_file2}']
exec(open('inference_gemma_sft.py').read())
"""
    
    # Запускаем процессы
    print(f"\n🚀 Запуск инференса на GPU {gpu1}...")
    env1 = os.environ.copy()
    env1['INFERENCE_GPU'] = gpu1
    env1['TOKENIZERS_PARALLELISM'] = 'false'
    env1['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Модифицируем inference_gemma_sft.py чтобы использовать правильный выходной файл
    # Вместо этого, создадим обертку
    cmd1 = [
        'python3', '-c', f"""
import os, sys
os.environ['INFERENCE_GPU'] = '{gpu1}'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.argv = ['inference_gemma_sft.py', '{checkpoint}', '{temp_file1}']
exec(open('inference_gemma_sft.py').read().replace('test_outputs.tsv', '{output_file1}').replace('dev_outputs', 'dev_outputs_part1'))
"""
    ]
    
    # Проще: запустим напрямую и переименуем выходной файл
    process1 = subprocess.Popen(
        ['python3', 'inference_gemma_sft.py', checkpoint, temp_file1],
        env=env1,
        stdout=open(f'inference_gpu{gpu1}.log', 'w'),
        stderr=subprocess.STDOUT
    )
    
    print(f"🚀 Запуск инференса на GPU {gpu2}...")
    env2 = os.environ.copy()
    env2['INFERENCE_GPU'] = gpu2
    env2['TOKENIZERS_PARALLELISM'] = 'false'
    env2['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    process2 = subprocess.Popen(
        ['python3', 'inference_gemma_sft.py', checkpoint, temp_file2],
        env=env2,
        stdout=open(f'inference_gpu{gpu2}.log', 'w'),
        stderr=subprocess.STDOUT
    )
    
    print(f"\n⏳ Ожидание завершения процессов...")
    print(f"   GPU {gpu1}: PID {process1.pid}")
    print(f"   GPU {gpu2}: PID {process2.pid}")
    print(f"   Логи: inference_gpu{gpu1}.log и inference_gpu{gpu2}.log")
    
    # Ждем завершения
    process1.wait()
    process2.wait()
    
    print(f"\n✅ Оба процесса завершены!")
    
    # Объединяем результаты
    print(f"Объединение результатов...")
    try:
        # Определяем имена выходных файлов (inference_gemma_sft.py создаст их автоматически)
        if 'test' in input_file.lower():
            # Скрипт создаст test_outputs.tsv, но нам нужно найти правильные файлы
            # Проверяем, какие файлы были созданы
            import glob
            part_files = sorted(glob.glob('*part*.tsv') + glob.glob('test_outputs*.tsv'))
            if len(part_files) >= 2:
                output_file1 = part_files[0]
                output_file2 = part_files[1]
            else:
                # Если файлы не найдены, используем стандартные имена
                output_file1 = "test_outputs.tsv"  # Будет перезаписан, нужно исправить
                output_file2 = "test_outputs.tsv"
                print(f"⚠️  Файлы не найдены автоматически, используем стандартные имена")
        
        # Читаем результаты (может потребоваться ручное указание файлов)
        # Для надежности, проверим логи
        print(f"Проверка созданных файлов...")
        import glob
        all_outputs = glob.glob('*outputs*.tsv')
        print(f"Найдено файлов: {all_outputs}")
        
        # Пробуем найти файлы по паттерну
        if os.path.exists('test_outputs.tsv'):
            # Если файл существует, значит один из процессов его создал
            # Нужно разделить результаты по ID
            df_temp = pd.read_csv('test_outputs.tsv', sep='\t')
            mid_point = len(df_temp) // 2
            df_out1 = df_temp.iloc[:mid_point].copy()
            df_out2 = df_temp.iloc[mid_point:].copy()
        else:
            # Пробуем найти файлы с part в имени
            part1_files = [f for f in all_outputs if 'part1' in f.lower() or 'gpu' + str(gpu1) in f]
            part2_files = [f for f in all_outputs if 'part2' in f.lower() or 'gpu' + str(gpu2) in f]
            
            if part1_files and part2_files:
                df_out1 = pd.read_csv(part1_files[0], sep='\t')
                df_out2 = pd.read_csv(part2_files[0], sep='\t')
            else:
                raise FileNotFoundError("Не удалось найти выходные файлы")
        
        df_out1 = pd.read_csv(output_file1, sep='\t')
        df_out2 = pd.read_csv(output_file2, sep='\t')
        
        # Объединяем
        df_final = pd.concat([df_out1, df_out2], ignore_index=True)
        
        # Убеждаемся, что ID правильные
        df_final['ID'] = range(len(df_final))
        
        # Сохраняем
        df_final.to_csv(final_output, sep='\t', index=False, encoding='utf-8')
        print(f"✅ Результаты сохранены в {final_output}")
        print(f"   Всего строк: {len(df_final)}")
        
        # Применяем rule-based детокс для финальной обработки
        print(f"\n🔧 Применение rule-based детоксификации...")
        try:
            from finalize_submission import apply_rule_based_detox, ensure_format
            df_final = apply_rule_based_detox(df_final)
            df_final = ensure_format(df_final)
            if df_final is not None:
                final_detox = final_output.replace('.tsv', '_detox.tsv')
                df_final.to_csv(final_detox, sep='\t', index=False, encoding='utf-8')
                print(f"✅ Rule-based детокс применён, результат в {final_detox}")
                
                # Создаём финальный архив
                print(f"\n📦 Создание submission архива...")
                from finalize_submission import create_submission_archive
                zip_file = create_submission_archive(final_detox)
                print(f"✅ Финальный архив: {zip_file}")
        except Exception as e:
            print(f"⚠️  Предупреждение: не удалось применить rule-based детокс: {e}")
            print(f"   Используется базовый результат: {final_output}")
        
        # Удаляем временные файлы
        os.remove(temp_file1)
        os.remove(temp_file2)
        os.remove(output_file1)
        os.remove(output_file2)
        print(f"✅ Временные файлы удалены")
        
    except Exception as e:
        print(f"❌ Ошибка при объединении: {e}")
        print(f"   Проверьте файлы {output_file1} и {output_file2} вручную")
        sys.exit(1)

if __name__ == "__main__":
    main()

