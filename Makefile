# Define variables for virtual environment and app
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Install dependencies and create virtual environment if not exists
install:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install Flask matplotlib numpy

# Run the Flask application
run:
	. venv/bin/activate && FLASK_ENV=development FLASK_APP=app.py flask run --port 3000

# Clean the virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies (Clean and Install)
reinstall: clean install