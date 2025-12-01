#!/bin/bash
# Полный пайплайн: инференс + rule-based детокс + создание submission архива

set -e

CHECKPOINT=${1:-"checkpoint-1594"}
INPUT_FILE=${2:-"test_inputs.tsv"}
GPU1=${3:-3}
GPU2=${4:-4}

echo "🚀 Полный пайплайн детоксификации"
echo "   Чекпоинт: $CHECKPOINT"
echo "   Входной файл: $INPUT_FILE"
echo "   GPU: $GPU1, $GPU2"
echo ""

# Шаг 1: Параллельный инференс
echo "📊 Шаг 1: Параллельный инференс на GPU $GPU1 и $GPU2..."
python3 run_inference_parallel.py "$CHECKPOINT" "$INPUT_FILE" "$GPU1" "$GPU2"

# Определяем выходной файл
if [[ "$INPUT_FILE" == *"test"* ]]; then
    INFERENCE_OUTPUT="test_outputs.tsv"
    FINAL_OUTPUT="test_outputs_final.tsv"
else
    INFERENCE_OUTPUT="dev_outputs_epoch2.tsv"
    FINAL_OUTPUT="dev_outputs_final.tsv"
fi

# Проверяем, что инференс завершился успешно
if [ ! -f "$INFERENCE_OUTPUT" ]; then
    echo "❌ Ошибка: файл $INFERENCE_OUTPUT не найден после инференса!"
    exit 1
fi

echo "✅ Инференс завершён: $INFERENCE_OUTPUT"
echo ""

# Шаг 2: Rule-based детокс
echo "🔧 Шаг 2: Применение rule-based детоксификации..."
python3 finalize_submission.py "$INPUT_FILE" "$FINAL_OUTPUT"

if [ ! -f "$FINAL_OUTPUT" ]; then
    echo "❌ Ошибка: файл $FINAL_OUTPUT не создан!"
    exit 1
fi

echo "✅ Rule-based детокс применён: $FINAL_OUTPUT"
echo ""

# Шаг 3: Проверка формата
echo "✅ Шаг 3: Проверка формата submission..."
python3 check_submission.py "$FINAL_OUTPUT"

echo ""
echo "🎉 Пайплайн завершён успешно!"
echo ""
echo "📦 Финальные файлы:"
echo "   - $FINAL_OUTPUT (TSV файл)"
echo "   - ${FINAL_OUTPUT%.tsv}_submission.zip (архив для submission)"
echo ""
echo "✅ Готово к отправке!"

