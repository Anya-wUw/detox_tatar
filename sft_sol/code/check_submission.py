#!/usr/bin/env python3
"""
Проверка формата файла для сабмита
"""
import pandas as pd
import sys

def check_submission(file_path):
    """Проверяет файл на соответствие требованиям"""
    print(f"Проверка файла: {file_path}\n")
    
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False
    
    # Проверка колонок
    required_cols = ['ID', 'tat_toxic', 'tat_detox1']
    print("Проверка колонок:")
    for col in required_cols:
        if col in df.columns:
            print(f"  ✅ {col}")
        else:
            print(f"  ❌ {col} - ОТСУТСТВУЕТ!")
            return False
    
    # Проверка порядка колонок
    if list(df.columns) != required_cols:
        print(f"\n⚠️  Порядок колонок: {list(df.columns)}")
        print(f"   Ожидаемый порядок: {required_cols}")
        print("   Переупорядочиваю...")
        df = df[required_cols]
        df.to_csv(file_path, sep='\t', index=False, encoding='utf-8')
        print("   ✅ Исправлено!")
    
    # Проверка пустых значений
    print(f"\nПроверка пустых значений:")
    empty_mask = df['tat_detox1'].isna() | (df['tat_detox1'].astype(str).str.strip() == "")
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        print(f"  ⚠️  Найдено {empty_count} пустых значений в tat_detox1")
        print("   Заменяю на оригинальный текст...")
        df.loc[empty_mask, 'tat_detox1'] = df.loc[empty_mask, 'tat_toxic']
        df.to_csv(file_path, sep='\t', index=False, encoding='utf-8')
        print("   ✅ Исправлено!")
    else:
        print(f"  ✅ Пустых значений нет")
    
    # Статистика
    print(f"\nСтатистика:")
    print(f"  Всего строк: {len(df)}")
    print(f"  Средняя длина tat_toxic: {df['tat_toxic'].str.len().mean():.1f}")
    print(f"  Средняя длина tat_detox1: {df['tat_detox1'].astype(str).str.len().mean():.1f}")
    
    # Примеры
    print(f"\nПримеры (первые 3):")
    for i in range(min(3, len(df))):
        print(f"\n  Пример {i+1} (ID={df.iloc[i]['ID']}):")
        print(f"    tat_toxic: {df.iloc[i]['tat_toxic'][:80]}...")
        print(f"    tat_detox1: {str(df.iloc[i]['tat_detox1'])[:80]}...")
    
    print(f"\n✅ Файл готов для сабмита!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python3 check_submission.py <файл.tsv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = check_submission(file_path)
    sys.exit(0 if success else 1)


