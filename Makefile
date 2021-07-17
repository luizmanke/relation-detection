PACKAGE = sources
STATUS_PREFIX = "\033[1;32m[+]\033[0m "
ATTENTION_PREFIX = "\033[1;36m[!]\033[0m "

help:
	@echo "usage: make <command>"
	@echo
	@echo "Code quality commands:"
	@echo "    check            Run all linters."
	@echo "    flake8           Run flake8."
	@echo "    mypy             Run mypy."

check: flake8 mypy

flake8:
	@printf $(STATUS_PREFIX); echo "LINT FLAKE8: STYLE CHECKING"
	flake8 $(PACKAGE)/

mypy:
	@printf $(STATUS_PREFIX); echo "LINT MYPY: TYPE CHECKING"
	mypy $(PACKAGE)/

.PHONY: check flake8 help mypy
