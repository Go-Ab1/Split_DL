#!/bin/bash

# --------- CONFIG ----------
PYTHON_ENV="$HOME/pytorch_env"        
REQ_FILE="requirements.txt"
MAIN_SCRIPT="./Scripts/main.py"
# ----------------------------


if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8+."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$PYTHON_ENV" ]; then
    echo "Virtual environment '$PYTHON_ENV' not found. Creating..."
    python3 -m venv "$PYTHON_ENV"
fi


echo "Activating virtual environment: $PYTHON_ENV"
source "$PYTHON_ENV/bin/activate"

# pip install --upgrade pip

# Install dependencies
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE..."
    pip install -r "$REQ_FILE"
else
    echo "requirements.txt not found. Installing default dependencies..."
    pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 \
                numpy==2.3.3 pandas==2.3.3 matplotlib==3.10.7 \
                seaborn==0.13.2 scikit-learn==1.7.2 tqdm==4.67.1
fi

# --------- RUN TASKS ---------
if [ $# -eq 0 ]; then
    echo "No tasks provided! Usage: ./run_tasks.sh train test quantize"
    exit 1
fi

# Run main.py with provided @tasks
python3 "$MAIN_SCRIPT" "$@"
