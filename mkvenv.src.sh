#! /bin/bash

PYTHON_VENV_NAME=venv

if [ ! -d $PYTHON_VENV_NAME ]; then
    python3 -m venv $PYTHON_VENV_NAME
fi

source $PYTHON_VENV_NAME/bin/activate
