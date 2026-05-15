.PHONY: help latest install list-tags

# Detect platform: linux | mac
PLAT := $(shell uname -s | tr '[:upper:]' '[:lower:]' | sed 's/darwin/mac/')

# Conda base packages (mirrors the conda section of environment.<plat>.yaml).
# Updated here when the yaml's non-pip section changes.
CONDA_BASE_linux := python=3.13 pip 'libblas=*=*openblas*' 'liblapack=*=*openblas*' llvm-openmp
CONDA_BASE_mac   := python=3.13 pip 'libblas=*=*openblas*'
CONDA_BASE       := $(CONDA_BASE_$(PLAT))

LATEST := $(shell git tag --list '20*.*.*' --sort=-v:refname | head -n1)

help:
	@echo "Platform detected: $(PLAT)"
	@echo "Latest tag:        $(LATEST)"
	@echo
	@echo "Targets:"
	@echo "  make latest             build env from latest tag ($(LATEST))"
	@echo "  make <tag>              build env from a specific tag, e.g. make 2026.05.1"
	@echo "  make install TAG=<tag>  same, explicit form"
	@echo "  make list-tags          list available tags"
	@echo
	@echo "Each target creates a conda env named torch-<tag>."

list-tags:
	@git fetch --tags --quiet
	@git tag --list '20*.*.*' --sort=-v:refname

latest:
	@if [ -z "$(LATEST)" ]; then \
		echo "No tags found. Run the workflow first (gh workflow run monthly-upgrade.yml)."; \
		exit 1; \
	fi
	@$(MAKE) install TAG=$(LATEST)

# Pattern target so 'make 2026.05.1' works.
20%:
	@$(MAKE) install TAG=20$*

install:
	@if [ -z "$(TAG)" ]; then echo "Usage: make install TAG=<tag>"; exit 1; fi
	@if [ -z "$(CONDA_BASE)" ]; then echo "Unsupported platform: $(PLAT)"; exit 1; fi
	@git fetch --tags --quiet
	@if ! git rev-parse --verify "$(TAG)" >/dev/null 2>&1; then \
		echo "Unknown tag: $(TAG)"; exit 1; \
	fi
	@if conda env list | awk '{print $$1}' | grep -qx "torch-$(TAG)"; then \
		echo "Env torch-$(TAG) already exists. Remove with: conda env remove -n torch-$(TAG)"; \
		exit 1; \
	fi
	@echo "Building torch-$(TAG) for $(PLAT) from tag $(TAG)"
	conda create -y -n torch-$(TAG) --override-channels -c conda-forge $(CONDA_BASE)
	git show $(TAG):lockfiles/requirements.$(PLAT).txt > /tmp/torch-$(TAG).$(PLAT).txt
	conda run -n torch-$(TAG) --no-capture-output pip install -r /tmp/torch-$(TAG).$(PLAT).txt
	conda run -n torch-$(TAG) --no-capture-output python scripts/test_env.py
	@echo
	@echo "Done. Activate with:  conda activate torch-$(TAG)"
