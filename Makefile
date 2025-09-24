# Use bash for &&, [[ ]], etc.
SHELL := bash
.ONESHELL:
.SILENT:

.DEFAULT_GOAL := help

UV ?= uv

.PHONY: help setup precommit lint format test test-cov clean clean-build clean-pyc

help: ## Show this help
	@awk 'BEGIN { FS=":.*##"; print "\nTargets:" } \
	     /^[a-zA-Z0-9_.-]+:.*##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' \
	     $(MAKEFILE_LIST)

# run at clone to prepare pre-commit hooks
setup: ## Install/refresh pre-commit hooks
	$(UV)x pre-commit install
	$(UV)x pre-commit autoupdate

precommit: ## Run pre-commit on all files
	$(UV)x pre-commit run --all-files

lint: ## Ruff lint (check only)
	$(UV) run ruff check .

format: ## Ruff format + lint
	$(UV) run ruff format .
	$(UV) run ruff check .

test: ## Run pytest
	$(UV) run pytest -vv -s -ra

cov: ## Run pytest with coverage (honors [tool.coverage.*])
	$(UV) run pytest -vv -s  --cov --cov-report=term-missing

clean: clean-build clean-pyc ## Remove build, pycache, coverage artifacts

clean-build:
	rm -rf build/ dist/ *.egg-info/

clean-pyc:
	# Safer than piping to xargs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache .coverage htmlcov
