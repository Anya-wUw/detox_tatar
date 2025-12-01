#!/usr/bin/env python3
"""
Rule-based детоксификация татарских текстов
Агрессивная очистка токсичности с сохранением структуры и смысла
"""
import pandas as pd
import re
from loguru import logger

logger.add("rule_based_detox.log", rotation="10 MB")

# Русский мат и производные (полный список)
RUSSIAN_PROFANITY = [
    r'\bблэт\b', r'\bблять\b', r'\bбля\b', r'\bблядь\b',
    r'\bнахуй\b', r'\bнахрен\b', r'\bнахрена\b', r'\bнахуя\b',
    r'\bхуй\b', r'\bхуйня\b', r'\bхуев\b', r'\bхуево\b',
    r'\bпиздец\b', r'\bпизда\b', r'\bпизд\w+\b',
    r'\bзаебал\b', r'\bзаебали\b', r'\bзаебись\b', r'\bзаеб\w+\b',
    r'\bебаный\b', r'\bебать\b', r'\bеб\w+\b',
    r'\bдолбаеб\b', r'\bдолбоеб\b',
    r'\bсука\b', r'\bсуки\b',
    r'\bмразь\b', r'\bмрази\b',
    r'\bпидор\b', r'\bпидорас\b',
    r'\bговно\b', r'\bговна\b',
    r'\bнихуя\b', r'\bнихуясе\b', r'\bнихуя\w+\b',
    r'\bахуенчик\b', r'\bахуен\w+\b',
]

# Татарские обсценные конструкции
TATAR_PROFANITY = [
    r'\bкутак\w*\b',      # кутак, кутаклар, кутагыз и т.д.
    r'\bкутар\w*\b',      # кутарабез, кутарасыз и т.д.
    r'\bкутаг\w*\b',      # кутагым, кутагыма и т.д.
    r'\bкутен\w*\b',      # кутен, кутеннар и т.д.
    r'\bсег\w+\b',        # сегим, сегеп, сегэсез, сегэргэ и т.д.
]

# Жёсткие угрозы (удаляется вся фраза)
THREAT_PATTERNS = [
    r'суеп\s+утерэм[^.]*',  # суеп утерэм ... (до конца предложения или до точки)
    r'утерэм\s+сине[^.]*',
    r'үтерәм\s+сине[^.]*',
]

# Замены для улучшения fluency
FLUENCY_REPLACEMENTS = {
    r'\bтинтәк\b': 'ялгышасың',
    r'\bахмак\b': 'ялгышасың',
    r'\bдурак\b': 'ялгышасың',
    r'\bхайваннарны\b': 'кешеләрне',
    r'\bхайван\b': 'кеше',
    r'\bхашарат\b': 'әйбер',
    r'\bнихуясе\b': 'ничек инде',
    r'\bахуенчик\b': 'бик яхшы',
}


def clean_punctuation(text: str) -> str:
    """Чистка избыточной пунктуации"""
    # Множественные восклицательные знаки
    text = re.sub(r'!{3,}', '!!', text)
    # Множественные вопросительные знаки
    text = re.sub(r'\?{3,}', '??', text)
    # Множественные точки
    text = re.sub(r'\.{3,}', '...', text)
    # Пробелы перед пунктуацией
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    # Пробелы в начале и конце
    text = text.strip()
    return text


def remove_profanity(text: str) -> str:
    """Удаление токсичных слов с сохранением структуры"""
    result = text
    
    # Удаление жёстких угроз (целиком)
    for pattern in THREAT_PATTERNS:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Удаление русского мата
    for pattern in RUSSIAN_PROFANITY:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Удаление татарских обсценных конструкций
    for pattern in TATAR_PROFANITY:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    
    # Замены для fluency
    for pattern, replacement in FLUENCY_REPLACEMENTS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Чистка пунктуации и пробелов
    result = clean_punctuation(result)
    
    # Если после очистки осталась пустая строка или только пробелы/знаки
    if not result or not result.strip() or len(result.strip()) < 2:
        # Возвращаем минимально очищенный оригинал
        result = text
        # Только удаляем самые явные маты
        for pattern in RUSSIAN_PROFANITY[:5]:  # Только самые частые
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        result = clean_punctuation(result)
        if not result or not result.strip():
            result = text  # В крайнем случае возвращаем оригинал
    
    return result


def detoxify_text(toxic_text: str) -> str:
    """
    Основная функция детоксификации
    Делает минимальные точечные изменения, сохраняя структуру
    """
    if pd.isna(toxic_text) or not isinstance(toxic_text, str):
        return str(toxic_text) if not pd.isna(toxic_text) else ""
    
    # Удаляем токсичные слова
    detoxified = remove_profanity(toxic_text)
    
    # Гарантируем, что результат не пустой
    if not detoxified or not detoxified.strip():
        # Fallback: минимальная очистка оригинального текста
        detoxified = toxic_text
        # Удаляем только самые явные маты
        for pattern in RUSSIAN_PROFANITY[:3]:
            detoxified = re.sub(pattern, '', detoxified, flags=re.IGNORECASE)
        detoxified = clean_punctuation(detoxified)
        if not detoxified or not detoxified.strip():
            detoxified = toxic_text
    
    return detoxified.strip()


def process_file(input_file: str, output_file: str):
    """Обработка TSV файла с применением rule-based детокса"""
    logger.info(f"Загрузка файла: {input_file}")
    df = pd.read_csv(input_file, sep='\t', dtype=str)
    
    logger.info(f"Загружено {len(df)} строк")
    
    # Проверяем наличие колонок
    if 'tat_toxic' not in df.columns:
        logger.error("Колонка 'tat_toxic' не найдена!")
        return
    
    if 'tat_detox1' not in df.columns:
        logger.info("Колонка 'tat_detox1' не найдена, будет создана")
        df['tat_detox1'] = ''
    
    # Создаём ID если его нет
    if 'ID' not in df.columns:
        logger.info("Колонка 'ID' не найдена, будет создана")
        df.insert(0, 'ID', range(len(df)))
    
    # Применяем rule-based детокс к tat_toxic
    logger.info("Применение rule-based детоксификации...")
    df['tat_detox1'] = df['tat_toxic'].apply(detoxify_text)
    
    # Гарантируем отсутствие пустых значений
    empty_count = df['tat_detox1'].isna().sum() + (df['tat_detox1'] == '').sum()
    if empty_count > 0:
        logger.warning(f"Найдено {empty_count} пустых значений, заполняем оригинальным текстом")
        mask = df['tat_detox1'].isna() | (df['tat_detox1'] == '')
        df.loc[mask, 'tat_detox1'] = df.loc[mask, 'tat_toxic']
    
    # Убеждаемся, что ID - строки
    df['ID'] = df['ID'].astype(str)
    
    # Сохраняем результат
    logger.info(f"Сохранение результата в {output_file}")
    df.to_csv(output_file, sep='\t', index=False)
    
    # Статистика
    logger.info(f"Обработано {len(df)} строк")
    logger.info(f"Пустых значений: {df['tat_detox1'].isna().sum()}")
    logger.info(f"Колонки: {list(df.columns)}")
    
    # Примеры
    logger.info("\nПримеры обработки:")
    for i in range(min(5, len(df))):
        original = df.iloc[i]['tat_toxic']
        detoxified = df.iloc[i]['tat_detox1']
        logger.info(f"\nПример {i+1}:")
        logger.info(f"  Оригинал: {original}")
        logger.info(f"  Детоксифицированный: {detoxified}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python3 rule_based_detox.py <input_file> [output_file]")
        print("Пример: python3 rule_based_detox.py test_outputs.tsv test_outputs_detox.tsv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.tsv', '_detox.tsv')
    
    logger.info("=== Rule-based детоксификация ===")
    process_file(input_file, output_file)
    logger.info("=== Обработка завершена ===")

