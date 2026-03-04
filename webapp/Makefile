SHELL := /bin/bash

.PHONY: help dev build preview stats lint clean install deps deploy

PROJECT ?= ogbench
DIST_DIR ?= dist
NPM ?= npm
PYTHON ?= python3
WRANGLER ?= npx --yes wrangler

help:
	@echo "Available targets:"
	@echo "  dev      - Start development server"
	@echo "  build    - Build for production (runs stats first to refresh Dataset Explorer data)"
	@echo "  preview  - Preview production build"
	@echo "  stats    - Regenerate public/data/stats.json from tutorials/stats/*.csv"
	@echo "  lint     - Typecheck (astro check)"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install dependencies"
	@echo "  deps     - Alias for install"
	@echo "  deploy   - Deploy to Cloudflare Pages via wrangler"
	@echo ""
	@echo "Config:"
	@echo "  PROJECT=$(PROJECT)"
	@echo "  DIST_DIR=$(DIST_DIR)"

dev: build
	$(NPM) run dev

stats:
	$(PYTHON) scripts/build_stats_from_csv.py

build: stats
	$(NPM) run build

preview:
	$(NPM) run preview

lint:
	$(NPM) run lint

clean:
	rm -rf "$(DIST_DIR)/" .astro/ node_modules/.vite/

install:
	$(NPM) install

deps: install

deploy: lint build
	$(WRANGLER) pages deploy "$(DIST_DIR)" --project-name "$(PROJECT)"
