# Variables
PYTHON = python3
PIP = pip3

# Commands
install:
	$(PIP) install -r requirements.txt
	cd frontend && npm install

run:
	# Start the backend Flask server
	$(PYTHON) backend/app.py &

	# Start the frontend React server
	cd frontend && npm start &

	# Wait to ensure both servers are running
	sleep 5

	# Test whether the app is running at localhost:3000
	curl -f http://localhost:3000 || echo "Failed to start application"

# Clean up all running processes
clean:
	kill `jobs -p` || echo "No jobs running"

# Testing GitHub Actions compatibility (add any further tests you have)
test:
	make install
	make run
	make clean
