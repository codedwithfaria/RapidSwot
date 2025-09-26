#!/bin/bash

# Start MongoDB (if not using external MongoDB)
mongod --fork --logpath /var/log/mongodb.log

# Start Redis (if not using external Redis)
redis-server --daemonize yes

# Start the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload