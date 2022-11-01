## About
Server implementation of sentence aligner using FastAPI and LASER toolkit.

## Usage
To install necessary packages for the script, run:

    pip install -r ./app/requirements.txt

Start the service:

    sh ./start.sh

Stop the service:

    sh ./stop.sh

Documentation is available at `http://localhost:8086/docs` (the server needs to be running).

## Docker
Build and run the image:

    docker build -t sentence-align-server .
    docker run -d -p 8086:8086 sentence-align-server:latest