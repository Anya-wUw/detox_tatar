#!/usr/bin/env python3
"""
Финальная обработка результатов с rule-based детоксом и создание submission архива
"""
import pandas as pd
import sys
import os
from rule_based_detox import detoxify_text, clean_punctuation
from loguru import logger

logger.add("finalize_submission.log", rotation="10 MB")


def merge_parts_if_needed(input_file: str) -> str:
    """Объединяет части файла если они есть"""
    base_name = input_file.replace('.tsv', '')
    part1 = f"{base_name}.part1.tsv"
    part2 = f"{base_name}.part2.tsv"
    
    if os.path.exists(part1) and os.path.exists(part2):
        logger.info(f"Обнаружены части файла: {part1} и {part2}")
        df1 = pd.read_csv(part1, sep='\t', dtype=str)
        df2 = pd.read_csv(part2, sep='\t', dtype=str)
        df = pd.concat([df1, df2], ignore_index=True)
        logger.info(f"Объединено {len(df1)} + {len(df2)} = {len(df)} строк")
        return df
    elif os.path.exists(input_file):
        logger.info(f"Используется файл: {input_file}")
        return pd.read_csv(input_file, sep='\t', dtype=str)
    else:
        logger.error(f"Файл {input_file} не найден!")
        return None


def apply_rule_based_detox(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет rule-based детокс к tat_toxic"""
    logger.info("Применение rule-based детоксификации к tat_toxic...")
    
    # Если tat_detox1 уже есть, используем его как основу, но применяем rule-based к tat_toxic
    # Пользователь хочет пройтись по оригинальному tat_toxic
    df['tat_detox1'] = df['tat_toxic'].apply(detoxify_text)
    
    # Гарантируем отсутствие пустых значений
    empty_mask = df['tat_detox1'].isna() | (df['tat_detox1'].astype(str).str.strip() == '')
    if empty_mask.sum() > 0:
        logger.warning(f"Найдено {empty_mask.sum()} пустых значений, заполняем оригинальным текстом")
        df.loc[empty_mask, 'tat_detox1'] = df.loc[empty_mask, 'tat_toxic']
    
    return df


def ensure_format(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует правильный формат submission"""
    # Создаём ID если его нет
    if 'ID' not in df.columns:
        logger.info("Создание колонки ID")
        df.insert(0, 'ID', range(len(df)))
    
    # Убеждаемся, что есть все колонки
    required_cols = ['ID', 'tat_toxic', 'tat_detox1']
    for col in required_cols:
        if col not in df.columns:
            if col == 'tat_toxic':
                logger.error("Колонка 'tat_toxic' обязательна!")
                return None
            elif col == 'tat_detox1':
                logger.error("Колонка 'tat_detox1' обязательна!")
                return None
    
    # Убеждаемся, что ID - строки
    df['ID'] = df['ID'].astype(str)
    
    # Убеждаемся, что нет пустых значений
    empty_mask = df['tat_detox1'].isna() | (df['tat_detox1'].astype(str).str.strip() == '')
    if empty_mask.sum() > 0:
        logger.warning(f"Заполнение {empty_mask.sum()} пустых значений")
        df.loc[empty_mask, 'tat_detox1'] = df.loc[empty_mask, 'tat_toxic']
    
    # Выбираем только нужные колонки в правильном порядке
    df = df[required_cols].copy()
    
    return df


def create_submission_archive(tsv_file: str, output_zip: str = None):
    """Создаёт финальный архив для submission"""
    import zipfile
    
    if output_zip is None:
        output_zip = tsv_file.replace('.tsv', '_submission.zip')
    
    # Имя файла внутри архива
    archive_name = "test_outputs.tsv" if 'test' in os.path.basename(tsv_file).lower() else "dev_outputs.tsv"
    
    logger.info(f"Создание zip-архива: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(tsv_file, arcname=archive_name)
    
    logger.info(f"✅ Zip-архив создан: {output_zip}")
    logger.info(f"   Размер: {os.path.getsize(output_zip) / 1024:.1f} KB")
    
    return output_zip


def main(input_file: str, output_file: str = None, create_zip: bool = True):
    """Основная функция"""
    logger.info("=== Финальная обработка для submission ===")
    
    # Объединяем части если нужно
    df = merge_parts_if_needed(input_file)
    if df is None:
        return False
    
    logger.info(f"Загружено {len(df)} строк")
    
    # Применяем rule-based детокс
    df = apply_rule_based_detox(df)
    
    # Гарантируем формат
    df = ensure_format(df)
    if df is None:
        return False
    
    # Определяем выходной файл
    if output_file is None:
        if 'test' in input_file.lower():
            output_file = "test_outputs_final.tsv"
        else:
            output_file = input_file.replace('.tsv', '_final.tsv')
    
    # Сохраняем результат
    logger.info(f"Сохранение результата в {output_file}")
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    
    # Статистика
    logger.info(f"\n✅ Обработано {len(df)} строк")
    logger.info(f"   Пустых значений: {df['tat_detox1'].isna().sum()}")
    logger.info(f"   Колонки: {list(df.columns)}")
    
    # Примеры
    logger.info("\nПримеры обработки:")
    for i in range(min(5, len(df))):
        original = df.iloc[i]['tat_toxic']
        detoxified = df.iloc[i]['tat_detox1']
        logger.info(f"\nПример {i+1}:")
        logger.info(f"  Оригинал: {original}")
        logger.info(f"  Детоксифицированный: {detoxified}")
    
    # Создаём архив
    if create_zip:
        zip_file = create_submission_archive(output_file)
        logger.info(f"\n✅ Финальный файл для submission: {zip_file}")
    
    logger.info("=== Обработка завершена ===")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python3 finalize_submission.py <input_file> [output_file] [--no-zip]")
        print("Пример: python3 finalize_submission.py test_outputs.tsv test_outputs_final.tsv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    create_zip = '--no-zip' not in sys.argv
    
    success = main(input_file, output_file, create_zip)
    sys.exit(0 if success else 1)

