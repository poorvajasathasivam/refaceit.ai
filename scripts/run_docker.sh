#!/bin/bash
set -e

echo "Building and starting Docker services..."
docker-compose up --build

# The script will continue when docker-compose is stopped
echo "Docker services stopped."
