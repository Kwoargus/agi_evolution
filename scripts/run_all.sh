# run_all.sh - Альтернативный способ запуска экспериментов

echo
"========================================"
echo
"ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ"
echo
"========================================"

# Определяем команду Python
if command - v python3 & > / dev / null; then
PYTHON_CMD = "python3"
elif command - v
python & > / dev / null;
then
PYTHON_CMD = "python"
else
echo
"❌ Python не найден!"
exit
1
fi

echo
"Используется: $PYTHON_CMD"

# Список экспериментов
EXPERIMENTS = (
    "scripts/test_gan.py"
    "scripts/compare_evolution_with_gan.py"
    "scripts/test_genome_improvement.py"
    "tests/test_gan_integration.py"
)

for SCRIPT in "${EXPERIMENTS[@]}"; do
echo
""
echo
"========================================"
echo
"Запуск: $SCRIPT"
echo
"========================================"

if [-f "$SCRIPT"]; then
$PYTHON_CMD
"$SCRIPT"
if [ $? -eq 0]; then
echo
"✅ $SCRIPT успешно выполнен"
else
echo
"❌ $SCRIPT завершился с ошибкой"
fi
else
echo
"❌ Файл не найден: $SCRIPT"
fi

echo
""
echo
"Нажмите Enter для продолжения..."
read
done

echo
"========================================"
echo
"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ"
echo
"========================================"