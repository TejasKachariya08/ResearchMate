PYTHON = python
STREAMLIT_FLAGS = --server.fileWatcherType none --browser.gatherUsageStats false --server.enableXsrfProtection false --server.address 0.0.0.0

.PHONY: help install format lint test check dev docker-up docker-down

help:
	@echo "ResearchMate Developer Commands:"
	@echo "  make install     - Install production dependencies"
	@echo "  make dev-install - Install all dependencies including dev tools"
	@echo "  make check       - Run environment & service verification"
	@echo "  make test        - Run automated test suite"
	@echo "  make format      - Format codebase with ruff"
	@echo "  make lint        - Lint codebase with ruff"
	@echo "  make dev         - Launch Streamlit application locally"
	@echo "  make docker-up   - Start full stack with Docker Compose"
	@echo "  make docker-down - Stop Docker Compose stack"

install:
	$(PYTHON) -m pip install -r requirements.txt

dev-install:
	$(PYTHON) -m pip install -r requirements-dev.txt

check:
	$(PYTHON) scripts/check_setup.py

test:
	$(PYTHON) -m pytest tests/ -v

format:
	$(PYTHON) -m ruff format src/ tests/ scripts/ app.py pages/

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/ app.py pages/

dev:
	$(PYTHON) -m streamlit run app.py $(STREAMLIT_FLAGS)

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
