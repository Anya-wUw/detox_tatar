#!/usr/bin/env python3
"""
Подготовка файла для сабмита (создание zip-архива)
"""
import pandas as pd
import zipfile
import os
import sys
from pathlib import Path

def prepare_submission(tsv_file, output_zip=None):
    """Подготавливает файл для сабмита"""
    
    if not os.path.exists(tsv_file):
        print(f"❌ Файл {tsv_file} не найден!")
        return False
    
    # Проверяем формат
    print(f"Проверка файла: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Проверка колонок
    required_cols = ['ID', 'tat_toxic', 'tat_detox1']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Отсутствуют обязательные колонки!")
        print(f"   Найдено: {list(df.columns)}")
        print(f"   Требуется: {required_cols}")
        return False
    
    # Убеждаемся, что порядок колонок правильный
    df = df[required_cols]
    
    # Проверка пустых значений
    empty_mask = df['tat_detox1'].isna() | (df['tat_detox1'].astype(str).str.strip() == "")
    if empty_mask.sum() > 0:
        print(f"⚠️  Найдено {empty_mask.sum()} пустых значений, заменяю на оригинал...")
        df.loc[empty_mask, 'tat_detox1'] = df.loc[empty_mask, 'tat_toxic']
    
    # Сохраняем исправленный файл во временный
    temp_file = tsv_file.replace('.tsv', '_fixed.tsv')
    df.to_csv(temp_file, sep='\t', index=False, encoding='utf-8')
    print(f"✅ Исправленный файл сохранен: {temp_file}")
    
    # Создаем zip-архив
    if output_zip is None:
        output_zip = tsv_file.replace('.tsv', '_submission.zip')
    
    # Имя файла внутри архива (должен быть один файл)
    # Для сабмита используем стандартное имя test_outputs.tsv
    if 'test' in os.path.basename(tsv_file).lower():
        archive_name = "test_outputs.tsv"
    elif 'dev' in os.path.basename(tsv_file).lower():
        archive_name = "dev_outputs.tsv"
    else:
        archive_name = os.path.basename(tsv_file)
    
    print(f"\nСоздание zip-архива: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(temp_file, arcname=archive_name)
    
    print(f"✅ Zip-архив создан: {output_zip}")
    print(f"   Размер: {os.path.getsize(output_zip) / 1024:.1f} KB")
    
    # Проверяем содержимое архива
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        files = zipf.namelist()
        print(f"   Файлов в архиве: {len(files)}")
        for f in files:
            print(f"     - {f}")
    
    # Удаляем временный файл
    os.remove(temp_file)
    print(f"\n✅ Готово! Файл для сабмита: {output_zip}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python3 prepare_submission.py <файл.tsv> [output.zip]")
        sys.exit(1)
    
    tsv_file = sys.argv[1]
    output_zip = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = prepare_submission(tsv_file, output_zip)
    sys.exit(0 if success else 1)

