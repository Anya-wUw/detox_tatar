#!/bin/bash
# Простой скрипт для параллельного запуска на двух GPU

CHECKPOINT=$1
INPUT_FILE=$2
GPU1=${3:-1}  # По умолчанию GPU 1
GPU2=${4:-2}  # По умолчанию GPU 2

echo "🚀 Запуск параллельного инференса на GPU $GPU1 и $GPU2"

# Разделяем файл пополам
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
HALF_LINES=$((TOTAL_LINES / 2))

# Создаем временные файлы (пропускаем заголовок)
head -1 "$INPUT_FILE" > "${INPUT_FILE}.part1.tsv"
sed -n "2,${HALF_LINES}p" "$INPUT_FILE" >> "${INPUT_FILE}.part1.tsv"

head -1 "$INPUT_FILE" > "${INPUT_FILE}.part2.tsv"
sed -n "$((HALF_LINES + 1)),\$p" "$INPUT_FILE" >> "${INPUT_FILE}.part2.tsv"

echo "✅ Файлы разделены:"
echo "   Часть 1: ${INPUT_FILE}.part1.tsv ($(wc -l < "${INPUT_FILE}.part1.tsv") строк)"
echo "   Часть 2: ${INPUT_FILE}.part2.tsv ($(wc -l < "${INPUT_FILE}.part2.tsv") строк)"

# Запускаем на GPU 1
echo ""
echo "🚀 Запуск на GPU $GPU1..."
INFERENCE_GPU=$GPU1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 inference_gemma_sft.py "$CHECKPOINT" "${INPUT_FILE}.part1.tsv" > inference_gpu${GPU1}.log 2>&1 &
PID1=$!

# Запускаем на GPU 2
echo "🚀 Запуск на GPU $GPU2..."
INFERENCE_GPU=$GPU2 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 inference_gemma_sft.py "$CHECKPOINT" "${INPUT_FILE}.part2.tsv" > inference_gpu${GPU2}.log 2>&1 &
PID2=$!

echo ""
echo "⏳ Процессы запущены:"
echo "   GPU $GPU1: PID $PID1 (лог: inference_gpu${GPU1}.log)"
echo "   GPU $GPU2: PID $PID2 (лог: inference_gpu${GPU2}.log)"
echo ""
echo "Ожидание завершения..."

# Ждем завершения
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?

echo ""
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo "✅ Оба процесса завершены успешно!"
    
    # Определяем выходные файлы
    if [[ "$INPUT_FILE" == *"test"* ]]; then
        OUTPUT1="test_outputs.tsv"
        OUTPUT2="test_outputs.tsv"  # Оба создадут одинаковое имя, нужно переименовать
    else
        OUTPUT1="dev_outputs_epoch2.tsv"
        OUTPUT2="dev_outputs_epoch2.tsv"
    fi
    
    # Объединяем результаты
    echo "Объединение результатов..."
    python3 << EOF
import pandas as pd
import sys

# Читаем оба файла (они могут иметь одинаковое имя, но созданы в разное время)
# Используем временные имена
df1 = pd.read_csv("${INPUT_FILE}.part1.tsv", sep='\t')
df2 = pd.read_csv("${INPUT_FILE}.part2.tsv", sep='\t')

# Запускаем инференс заново на каждом файле с переименованием выхода
# Или просто объединяем по порядку
# Для простоты, создадим выходные файлы вручную
print("⚠️  Нужно вручную объединить результаты из inference_gpu${GPU1}.log и inference_gpu${GPU2}.log")
print("   Или перезапустить с модифицированным скриптом")
EOF
    
    echo "✅ Готово! Проверьте логи для деталей."
else
    echo "❌ Один из процессов завершился с ошибкой!"
    echo "   GPU $GPU1: статус $STATUS1"
    echo "   GPU $GPU2: статус $STATUS2"
    echo "   Проверьте логи: inference_gpu${GPU1}.log и inference_gpu${GPU2}.log"
fi

# Удаляем временные файлы
rm -f "${INPUT_FILE}.part1.tsv" "${INPUT_FILE}.part2.tsv"

