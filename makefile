VENV_NAME = myvenv

venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || virtualenv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install -r requirements.txt
	$(VENV_NAME)/bin/python3 setup.py develop
	$(VENV_NAME)/bin/ipython kernel install --user --name=$(VENV_NAME)
	touch $(VENV_NAME)/bin/activate
