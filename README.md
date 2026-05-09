# Titanic Project

## Структура
- `configs/default.yaml` — все настройки запуска
- `main.py` — единая точка входа
- `src/` — модули пайплайна
- `notebooks/titanic_eda.ipynb` — EDA и аналитические выводы
- `artifacts/` — результаты запуска

## Как запустить
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
make run
