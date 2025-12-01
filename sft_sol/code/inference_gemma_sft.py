#!/usr/bin/env python3
"""
Инференс обученной Gemma модели на dev_inputs.tsv
Генерирует детоксифицированные тексты для соревнования
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger
from tqdm import tqdm
import os

logger.add("inference_gemma_sft.log", rotation="10 MB")

# Конфигурация
BASE_MODEL = "google/gemma-2b"
LORA_MODEL_PATH = "gemma_detox_sft"  # Путь к обученной LoRA модели (или checkpoint-XXX)
INPUT_FILE = "dev_inputs.tsv"
OUTPUT_FILE = "dev_outputs.tsv"
MAX_LENGTH = 256
BATCH_SIZE = 4
MAX_NEW_TOKENS = 96  # Уменьшено для избежания повторений
NUM_GENERATION_ATTEMPTS = 3  # Увеличено количество попыток для лучшего результата

# Можно указать конкретный чекпоинт и входной файл через аргументы командной строки
import sys
USE_FILTER = True  # Использовать фильтр токсичных слов

if len(sys.argv) > 1:
    checkpoint_name = sys.argv[1]
    LORA_MODEL_PATH = f"gemma_detox_sft/{checkpoint_name}"
    logger.info(f"Используется чекпоинт: {checkpoint_name}")

# Второй аргумент - входной файл (опционально)
if len(sys.argv) > 2:
    INPUT_FILE = sys.argv[2]
    logger.info(f"Используется входной файл: {INPUT_FILE}")
    # Определяем выходной файл на основе входного
    if "test" in INPUT_FILE.lower():
        if "part1" in INPUT_FILE.lower():
            OUTPUT_FILE = "test_outputs_part1.tsv"
        elif "part2" in INPUT_FILE.lower():
            OUTPUT_FILE = "test_outputs_part2.tsv"
        else:
            OUTPUT_FILE = "test_outputs.tsv"
    elif "dev" in INPUT_FILE.lower():
        if "part1" in INPUT_FILE.lower():
            OUTPUT_FILE = "dev_outputs_part1.tsv"
        elif "part2" in INPUT_FILE.lower():
            OUTPUT_FILE = "dev_outputs_part2.tsv"
        elif "1594" in checkpoint_name:
            OUTPUT_FILE = "dev_outputs_epoch2.tsv"
        elif "797" in checkpoint_name:
            OUTPUT_FILE = "dev_outputs_epoch1.tsv"
        else:
            OUTPUT_FILE = f"dev_outputs_{checkpoint_name}.tsv"
    else:
        OUTPUT_FILE = INPUT_FILE.replace("_inputs.tsv", "_outputs.tsv")
    logger.info(f"Выходной файл: {OUTPUT_FILE}")
elif len(sys.argv) > 1:
    # Если только чекпоинт указан, используем dev_inputs.tsv
    checkpoint_name = sys.argv[1]
    if "1594" in checkpoint_name:
        OUTPUT_FILE = "dev_outputs_epoch2.tsv"
    elif "797" in checkpoint_name:
        OUTPUT_FILE = "dev_outputs_epoch1.tsv"
    else:
        OUTPUT_FILE = f"dev_outputs_{checkpoint_name}.tsv"

# Детальный системный промпт для улучшения качества детоксификации
TATAR_PROMPT = """You are a text detoxification assistant for the Tatar language.
Your goal is to rewrite toxic Tatar sentences into non-toxic Tatar, while preserving the main meaning and keeping the text natural and fluent.

Strict rules:

Language:
- Input and output must be in Tatar (tt).
- Do not translate to Russian or English.

Detoxification:
Remove or soften:
- Оскорбления, маты, грубые выражения (в т.ч. русские маты внутри татарского текста).
- Угрозы физического насилия и унижения.
- Жёсткие обращения к человеку как к «хайван», «ахмак», «тинтәк» и т.п.

При этом:
- Сохраняй основную мысль (например: «я с тобой не согласен», «мне не нравится», «я зол», «это очень плохо»).
- Можно заменять грубые выражения на мягкие:
  «тинтәк», «ахмак» → «ялгышасың», «дөрес фикер түгел»
  грубые ругательства → нейтральные фразы «миңа ошамый», «бу начар», «бу дөрес түгел».

Стилистика:
- Сохраняй стиль и эмоциональность, но без оскорблений и мата.
- Разрешены: Усилители эмоций («бик начар», «бер дә ошамый», «боелмый торган хәл»).
- Разрешены: Мягкая критика и несогласие.
- Нельзя: Добавлять новые оскорбления, усиливать агрессию.

Формат ответа:
- На каждый входной текст возвращай только детоксифицированный вариант без пояснений, без перевода, без комментариев.
- Не изменяй тему и факты: если говорилось про футбол, не замещай на другую тему.

Примеры:

Input (toxic): «Син нинди тинтәк кеше!»
Output (detox): «Син ялгышасың!»

Input (toxic): «Утырасың да күңел ачасың, ахмак!!!!! сүзләр юк!»
Output (detox): «Утырасың да күңел ачасың!!!!! сүзләр юк!»

Input (toxic): «Андый хайваннарны кабер генә төзәтә!»
Output (detox): «Андый кешеләрне бернәрсә дә төзәтми!»

Remember: your main objective is to remove toxicity while preserving the core meaning of the original Tatar text.

Now detoxify this text:
Токсик: """

# Список токсичных слов для фильтрации
SAFE_REMOVE_TOKENS = [
    # русские матные вставки, часто встречающиеся в татарской речи
    "бля", "бляя", "блят", "блять", "блядь", "блэт", "блэ", "блт",
    "нах", "нахуй", "нахрен", "нахер", "нахуя",
    # пизд-корень (эмоциональные междометия, без смысла)
    "пизда", "пезда", "пездос", "пиздос", "пиздюк", "пездюк",
    "пиздец", "пездэц", "пездес", "пиздосик", "пиздобратия",
    "пиздобол", "пиздабол",
    # хуе-корень (ругательные вставки-уничижения)
    "хуеплет", "хуеплёт", "хуесос", "хуесослар", "хуепачмак",
    "хуебет", "хуебоз", "хуйлоп", "хуйло", "хуйня",
    "хуита", "хуита", "хуёв", "хуевина",
    # оскорбления, не несущие фактического смысла
    "пидор", "пидарас", "пидорас", "пидар", "пидрилла",
    "пидарок", "пидр", "пидрия", "педик", "гомик",
    # мягкая и бессмысленная ругань
    "долбаеб", "долбоеб", "долбоёб", "дебил", "идиот",
    "чмо", "чмошник",
    # скатологические ругательства (как междометия)
    "срандель", "сратый", "говнюк", "гавнюк",
    # «жопа»-ругательства, не влияющие на семантику
    "жопа", "жопоротый", "жополиза", "жопашник",
    # эмоциональные междометия, не несущие смысла
    "әттәгенә", "әттәгенәһе", "әпәт", "әпәәт",
    # длинные ругательные обороты, пустые по смыслу
    "сука", "сучара", "мразь", "тварь",
    # дополнительные негативные слова
    "заебали", "заебал", "заебись", "ебанутый", "ебанулся",
    "ебануть", "ебанула", "ебануло", "ебанули",
    "ебан", "ебать", "ебаный", "ебаная", "ебаное",
    "ебло", "еблан", "ебла", "ебланка"
]

def filter_toxic_words(text):
    """Удаляет токсичные слова из текста с умной заменой"""
    if not text or pd.isna(text):
        return text
    
    # Словарь замены токсичных слов на нейтральные синонимы
    replacements = {
        "блэт": "", "бля": "", "блять": "", "блядь": "",
        "хуй": "", "хуйня": "", "хуйло": "",
        "пизда": "", "пиздец": "", "пиздос": "",
        "ебан": "", "ебать": "", "ебаный": "",
        "заебали": "йомшарттылар", "заебал": "йомшартты",
        "долбаеб": "ақылсыз", "долбоеб": "ақылсыз",
        "сука": "", "мразь": "", "тварь": "",
    }
    
    text_lower = text.lower()
    original_words = text.split()
    cleaned_words = []
    
    for word in original_words:
        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
        # Проверяем точное совпадение
        if word_lower in SAFE_REMOVE_TOKENS:
            # Пропускаем токсичное слово
            continue
        # Проверяем, содержит ли слово токсичный токен
        contains_toxic = any(token in word_lower for token in SAFE_REMOVE_TOKENS if len(token) > 3)
        if contains_toxic:
            # Пытаемся заменить, если есть замена
            replaced = False
            for toxic, replacement in replacements.items():
                if toxic in word_lower:
                    if replacement:
                        cleaned_words.append(replacement)
                    replaced = True
                    break
            if not replaced:
                # Если нет замены, просто пропускаем
                continue
        else:
            # Сохраняем слово
            cleaned_words.append(word)
    
    result = ' '.join(cleaned_words)
    # Убираем множественные пробелы
    result = ' '.join(result.split())
    
    return result

def format_prompt(toxic_text):
    """Форматирует промпт для Gemma с детальным системным промптом"""
    instruction = f"{TATAR_PROMPT}{toxic_text}\nOutput (detox):"
    return instruction

def load_model():
    """Загружает базовую модель и LoRA веса"""
    # Используем конкретный GPU для инференса
    inference_gpu = os.environ.get("INFERENCE_GPU", "auto")
    if inference_gpu != "auto":
        device_map = {"": int(inference_gpu)}
        logger.info(f"Используется GPU {inference_gpu} для инференса")
    else:
        device_map = "auto"
        logger.info("Использование auto device_map для инференса")
    
    logger.info(f"Загрузка базовой модели: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загружаем модель
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    
    # Загрузка LoRA весов
    if os.path.exists(LORA_MODEL_PATH):
        logger.info(f"Загрузка LoRA весов из {LORA_MODEL_PATH}")
        try:
            model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
            model = model.merge_and_unload()  # Объединяем веса для инференса
        except Exception as e:
            logger.error(f"Ошибка загрузки LoRA: {e}")
            logger.warning("Используем базовую модель")
            model = base_model
    else:
        logger.warning(f"LoRA модель не найдена в {LORA_MODEL_PATH}, используем базовую модель")
        model = base_model
    
    model.eval()
    logger.info("Модель загружена и готова к инференсу")
    
    return tokenizer, model

def clean_generated_text(text, original_text):
    """Очищает сгенерированный текст от повторений и промптов"""
    if not text:
        return original_text
    
    # Убираем промпт, если он есть
    if TATAR_PROMPT in text:
        text = text.replace(TATAR_PROMPT, "").strip()
    
    # Убираем повторяющиеся строки (берем первую уникальную строку)
    lines = text.split('\n')
    seen = set()
    unique_lines = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            unique_lines.append(line_clean)
    
    # Берем первую значимую строку (не пустую, не слишком короткую)
    result = None
    for line in unique_lines:
        # Убираем кавычки
        line = line.strip('"').strip("'").strip()
        if len(line) >= 3:
            result = line
            break
    
    # Если ничего не подошло, берем первую непустую строку
    if not result and unique_lines:
        result = unique_lines[0].strip('"').strip("'").strip()
    
    # Если все еще пусто, используем оригинал
    if not result or len(result) < 3:
        result = original_text
    
    # Убираем лишние пробелы и нормализуем
    result = ' '.join(result.split())
    
    # Убираем повторяющиеся слова (если одно слово повторяется 3+ раза подряд)
    words = result.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word.lower() == prev_word:
            repeat_count += 1
            if repeat_count < 2:  # Разрешаем максимум 2 повторения
                cleaned_words.append(word)
        else:
            repeat_count = 0
            cleaned_words.append(word)
        prev_word = word.lower()
    
    result = ' '.join(cleaned_words)
    
    return result

def validate_result(result, original_text):
    """Валидирует результат детоксификации"""
    if not result or len(result.strip()) < 3:
        return False
    
    # Проверяем, что результат не слишком отличается от оригинала по длине
    len_ratio = len(result) / len(original_text) if len(original_text) > 0 else 1.0
    if len_ratio < 0.3 or len_ratio > 3.0:
        return False
    
    # Проверяем наличие токсичных слов
    result_lower = result.lower()
    toxic_count = sum(1 for token in SAFE_REMOVE_TOKENS if token in result_lower)
    if toxic_count > 0:
        return False
    
    return True

def generate_detoxified(tokenizer, model, toxic_text, max_new_tokens=128, num_attempts=3):
    """Генерирует детоксифицированный текст с несколькими попытками"""
    prompt = format_prompt(toxic_text)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(model.device)
    
    prompt_length = inputs['input_ids'].shape[1]
    
    best_result = None
    best_score = float('inf')  # Меньше токсичных слов = лучше
    
    for attempt in range(num_attempts):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6 + attempt * 0.1,  # Разные температуры для разнообразия
                top_p=0.85,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Увеличенный штраф за повторения
                no_repeat_ngram_size=3,  # Запрет на повторение 3-грамм
                length_penalty=1.1,  # Предпочтение более коротким ответам
            )
        
        # Декодируем только сгенерированную часть (после промпта)
        generated_ids = outputs[0][prompt_length:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Очищаем от повторений и промптов
        generated = clean_generated_text(generated, toxic_text)
        
        # Применяем фильтр токсичных слов
        if USE_FILTER:
            generated_filtered = filter_toxic_words(generated)
            if generated_filtered and len(generated_filtered.strip()) >= 3:
                generated = generated_filtered
        
        # Валидируем результат
        is_valid = validate_result(generated, toxic_text)
        
        # Оцениваем качество (считаем токсичные слова)
        toxic_count = sum(1 for token in SAFE_REMOVE_TOKENS if token in generated.lower())
        
        # Дополнительная оценка: проверяем сохранение смысла (общие слова)
        original_words = set(toxic_text.lower().split())
        result_words = set(generated.lower().split())
        common_words = len(original_words & result_words)
        similarity_score = common_words / max(len(original_words), 1)
        
        # Комбинированная оценка: меньше токсичных слов + больше сохраненных слов
        quality_score = toxic_count * 10 - similarity_score * 5  # Меньше = лучше
        
        # Если это лучший результат
        if generated and len(generated.strip()) >= 3:
            if best_result is None or (is_valid and quality_score < best_score) or (not is_valid and best_score == float('inf')):
                best_result = generated
                best_score = quality_score
        
        # Если нашли идеальный результат (валидный и 0 токсичных слов), останавливаемся
        if is_valid and best_score <= 0:
            break
    
    # Если все попытки неудачны, используем оригинал с агрессивным фильтром
    if not best_result or len(best_result.strip()) < 3 or not validate_result(best_result, toxic_text):
        # Применяем агрессивную фильтрацию к оригиналу
        best_result = filter_toxic_words(toxic_text)
        
        # Если после фильтрации осталось мало текста, пытаемся восстановить смысл
        if not best_result or len(best_result.strip()) < 3:
            # Берем оригинал и просто удаляем токсичные слова, сохраняя структуру
            words = toxic_text.split()
            filtered_words = []
            for word in words:
                word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                if word_lower not in SAFE_REMOVE_TOKENS and not any(token in word_lower for token in SAFE_REMOVE_TOKENS if len(token) > 3):
                    filtered_words.append(word)
            best_result = ' '.join(filtered_words) if filtered_words else toxic_text
    
    # Финальная валидация и очистка
    if best_result:
        best_result = best_result.strip()
        # Убираем лишние пробелы
        best_result = ' '.join(best_result.split())
        # Применяем финальный фильтр
        best_result = filter_toxic_words(best_result)
    
    return best_result if best_result and len(best_result.strip()) >= 3 else toxic_text
    
    # Декодируем только сгенерированную часть (после промпта)
    generated_ids = outputs[0][prompt_length:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Очищаем от повторений и промптов
    generated = clean_generated_text(generated, toxic_text)
    
    # Если пусто или слишком коротко, возвращаем оригинал
    if not generated or len(generated.strip()) < 3:
        generated = toxic_text
    
    # Применяем фильтр токсичных слов
    if USE_FILTER:
        generated_filtered = filter_toxic_words(generated)
        # Если после фильтрации осталось достаточно текста, используем отфильтрованную версию
        if generated_filtered and len(generated_filtered.strip()) >= 3:
            generated = generated_filtered
        # Если после фильтрации осталось мало текста, но оригинал был токсичным, оставляем отфильтрованную версию
        elif len(generated.strip()) >= 3:
            # Оставляем как есть, если фильтрация удалила слишком много
            pass
    
    return generated

def process_batch(tokenizer, model, texts):
    """Обрабатывает батч текстов"""
    results = []
    for text in texts:
        detoxified = generate_detoxified(
            tokenizer, 
            model, 
            text, 
            max_new_tokens=MAX_NEW_TOKENS,
            num_attempts=NUM_GENERATION_ATTEMPTS
        )
        results.append(detoxified)
    return results

def main():
    logger.info("=== Начало инференса ===")
    
    # Загрузка модели
    tokenizer, model = load_model()
    
    # Загрузка входных данных
    logger.info(f"Загрузка {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep='\t')
    logger.info(f"Загружено {len(df)} примеров")
    
    # Проверка колонок
    if 'tat_toxic' not in df.columns:
        logger.error(f"Колонка 'tat_toxic' не найдена в {INPUT_FILE}")
        logger.info(f"Доступные колонки: {df.columns.tolist()}")
        return
    
    # Если есть колонка ID, сохраняем её
    has_id = 'ID' in df.columns
    if not has_id:
        logger.info("Колонка ID не найдена, будет создана")
        df.insert(0, 'ID', range(len(df)))
    
    # Генерация детоксифицированных текстов
    logger.info("Генерация детоксифицированных текстов...")
    detoxified_texts = []
    
    for idx in tqdm(range(0, len(df), BATCH_SIZE), desc="Обработка"):
        batch = df.iloc[idx:idx+BATCH_SIZE]
        batch_texts = batch['tat_toxic'].tolist()
        
        batch_results = process_batch(tokenizer, model, batch_texts)
        detoxified_texts.extend(batch_results)
    
    # Заполнение результатов
    df['tat_detox1'] = detoxified_texts
    
    # Проверка на пустые значения (заменяем на оригинал)
    empty_mask = df['tat_detox1'].isna() | (df['tat_detox1'].astype(str).str.strip() == "")
    df.loc[empty_mask, 'tat_detox1'] = df.loc[empty_mask, 'tat_toxic']
    
    logger.info(f"Пустых значений: {empty_mask.sum()}")
    
    # Убеждаемся, что все необходимые колонки есть в правильном порядке
    required_cols = ['ID', 'tat_toxic', 'tat_detox1']
    
    # Проверяем и создаем недостающие колонки
    if 'ID' not in df.columns:
        df.insert(0, 'ID', range(len(df)))
    
    # Убеждаемся, что tat_toxic сохранен (не изменяем его)
    if 'tat_toxic' not in df.columns:
        logger.error("Колонка tat_toxic отсутствует!")
        return
    
    # Убеждаемся, что tat_detox1 заполнена
    if 'tat_detox1' not in df.columns:
        logger.error("Колонка tat_detox1 отсутствует!")
        return
    
    # Финальная проверка на пустые значения в tat_detox1
    empty_mask = df['tat_detox1'].isna() | (df['tat_detox1'].astype(str).str.strip() == "")
    if empty_mask.sum() > 0:
        logger.warning(f"Найдено {empty_mask.sum()} пустых значений, заменяем на оригинал")
        df.loc[empty_mask, 'tat_detox1'] = df.loc[empty_mask, 'tat_toxic']
    
    # Выбираем только нужные колонки в правильном порядке
    output_df = df[required_cols].copy()
    
    # Проверяем размер файла
    logger.info(f"Размер входного файла: {len(df)} строк")
    logger.info(f"Размер выходного файла: {len(output_df)} строк")
    
    if len(output_df) != len(df):
        logger.warning(f"Размеры не совпадают! Входной: {len(df)}, Выходной: {len(output_df)}")
    
    # Сохранение
    logger.info(f"Сохранение результатов в {OUTPUT_FILE}")
    output_df.to_csv(OUTPUT_FILE, sep='\t', index=False, encoding='utf-8')
    
    logger.info(f"✅ Файл сохранен: {OUTPUT_FILE}")
    logger.info(f"   Колонки: {output_df.columns.tolist()}")
    logger.info(f"   Строк: {len(output_df)}")
    
    # Статистика
    logger.info("\n=== Статистика ===")
    logger.info(f"Обработано примеров: {len(df)}")
    logger.info(f"Средняя длина оригиналов: {df['tat_toxic'].str.len().mean():.1f}")
    logger.info(f"Средняя длина детоксифицированных: {df['tat_detox1'].str.len().mean():.1f}")
    
    # Примеры
    logger.info("\n=== Примеры ===")
    for i in range(min(3, len(df))):
        logger.info(f"\nПример {i+1}:")
        logger.info(f"Оригинал: {df.iloc[i]['tat_toxic'][:100]}...")
        logger.info(f"Детоксифицированный: {df.iloc[i]['tat_detox1'][:100]}...")
    
    logger.info("=== Инференс завершен ===")

if __name__ == "__main__":
    main()

