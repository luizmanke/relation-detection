IMAGE = relation-detection
PACKAGE = sources
PWD := $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)
USER_ID := $(UID):$(GID)
DOCKER_RUN = docker run -t \
						--rm \
						--user $(USER_ID)
DOCKER_JUPYTER := jupyter/datascience-notebook
STATUS_PREFIX = "\033[1;32m[+]\033[0m "
ATTENTION_PREFIX = "\033[1;36m[!]\033[0m "

help:
	@echo "usage: make <command>"
	@echo
	@echo "Pipeline commands:"
	@echo "    run-docker       Run project in a docker container."
	@echo
	@echo "Debugging and code quality commands:"
	@echo "    build            Init environment: Build docker image from Dockerfile."
	@echo "    check            Run all tests and linters."
	@echo "    tests            Run unit-tests."
	@echo "    shell            Enter into a shell (sh) of the docker container for debugging purposes."
	@echo
	@echo "Jupyter commands:"
	@echo "    notebook   		Launch jupyterlab with the software installed as library."

run-docker: build
	@printf $(STATUS_PREFIX); echo "DOCKER RUN"
	$(DOCKER_RUN) $(IMAGE) /bin/bash -c "python -m sources.main"

shell: build
	@printf $(STATUS_PREFIX); echo "OPEN SHELL"
	$(DOCKER_RUN) -it --entrypoint="sh" $(IMAGE)

check: build flake8 mypy tests

build:
	@printf $(STATUS_PREFIX); echo "BUILD DOCKER IMAGE"
	docker build -t $(IMAGE) .

flake8:
	@printf $(STATUS_PREFIX); echo "LINT FLAKE8: STYLE CHECKING"
	$(DOCKER_RUN) --entrypoint="flake8" $(IMAGE) \
				  $(PACKAGE)/

mypy:
	@printf $(STATUS_PREFIX); echo "LINT MYPY: TYPE CHECKING"
	$(DOCKER_RUN) --entrypoint="mypy" $(IMAGE) \
				  --ignore-missing-imports \
				  --strict-optional \
				  $(PACKAGE)/

notebook:
	@printf $(STATUS_PREFIX); echo "PREPARING JUPYTER NOTEBOOK ENVIRONMENT"
	@docker kill jupyter 2> /dev/null || echo No jupyter running now
	docker run -d \
				--rm \
				-p 8888:8888 \
			   	--name="jupyter" \
			   	--user root \
		       	-e NB_UID=$(UID) -e NB_GID=$(GID) \
				-v $(PWD):/home/jovyan/work \
				$(DOCKER_JUPYTER) \
			   	start.sh jupyter notebook --NotebookApp.token=''
	docker exec -t -d jupyter chown -R $(USER_ID) /home/jovyan/work/.
	@printf $(ATTENTION_PREFIX); echo "JupyterNotebook: http://127.0.0.1:8888"

.PHONY: build check flake8 help mypy notebook run shell tests
