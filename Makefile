PACKAGE = relation_detection
STATUS_PREFIX = "\033[1;32m[+]\033[0m "
ATTENTION_PREFIX = "\033[1;36m[!]\033[0m "

check: flake8 mypy test

flake8:
	@printf $(STATUS_PREFIX); echo "LINT FLAKE8: STYLE CHECKING"
	flake8 $(PACKAGE)/

mypy:
	@printf $(STATUS_PREFIX); echo "LINT MYPY: TYPE CHECKING"
	mypy $(PACKAGE)/

test:
	@printf $(STATUS_PREFIX); echo "RUN UNIT TESTS"
	pytest

.PHONY: check flake8 mypy test
