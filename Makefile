PYTHON = python3
ARGS = ?hdgcn
MAIN = main.py
SETUP = setup.py

install:
	$(PYTHON) $(SETUP) install

run:
	$(PYTHON) $(MAIN) $(ARGS)

clean:
	rm -rf __pycache__ *.pyc *.pyo

clean_dev:
	rm -rf __pycache__ *.pyc *.pyo 
	rm -rf build dist poseactionx.egg-info
	
.PHONY: install run clean