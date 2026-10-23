import sys
from pathlib import Path
from agent.baseline_model import run_fast_baseline


def main():
    if len(sys.argv) < 2:
        print("Использование: python solution.py <путь_к_task.md>")
        sys.exit(1)

    task_file = Path(sys.argv[1]).resolve()
    if not task_file.exists():
        print(f"Ошибка: файл {task_file} не найден")
        sys.exit(1)

    task_dir = task_file.parent
    print(f"Запуск пайплайна для задачи в каталоге: {task_dir}")

    run_fast_baseline(task_dir, task_file)


if __name__ == "__main__":
    main()