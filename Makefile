PYTHON ?= python
CONFIG ?= configs/default.yaml

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

eda:
	$(PYTHON) -m jupyter notebook notebooks/titanic_eda.ipynb

clean:
	rm -f artifacts/*.csv artifacts/*.json artifacts/*.png

archive:
	zip -r titanic-mentor-ready.zip . \
		-x ".venv/*" \
		-x "__pycache__/*" \
		-x ".pytest_cache/*" \
		-x ".git/*"