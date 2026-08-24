#!/bin/bash
# setup_env.sh - Create virtual environment and install dependencies

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate: source venv/bin/activate"