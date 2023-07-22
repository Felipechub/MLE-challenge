#!/bin/bash

# Get the port from the environment and provide a default
PORT=${PORT:-8080}

# Start uvicorn with the obtained port
exec uvicorn challenge.api:app --host 0.0.0.0 --port $PORT
