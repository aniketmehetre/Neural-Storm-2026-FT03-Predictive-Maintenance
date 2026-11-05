.PHONY: install test api dashboard demo benchmark clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run src/dashboard/app.py

frontend:
	npm install && npm run dev

benchmark:
	$(PYTHON) -m src.benchmark --quick

demo:
	@echo "Starting Engines Health Monitor in Offline Demo Mode..."
	$(PYTHON) -m src.benchmark --quick
	$(PYTHON) -m pytest tests/ -v
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info

