#! /bin/bash

PYTHON_VENV_NAME=StockPrediction
PYTHON_VERSION=3.6.6

pyenv virtualenvs | grep $PYTHON_VENV_NAME > /dev/null
found=$?

if [ $found -ne 0 ]; then
    pyenv virtualenv $PYTHON_VERSION $PYTHON_VENV_NAME
fi

pyenv activate $PYTHON_VENV_NAME
