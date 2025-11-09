#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"
PYTHON_ENV="${SCRIPT_DIR}/pydevs"
FOURASTRO_MODULE_PATH="${SCRIPT_DIR}/fourastro"

if [ ! -d "${PYTHON_ENV}" ]; then
    mkdir -p "${PYTHON_ENV}"
    python3 -m venv "${PYTHON_ENV}"    
fi

source "${PYTHON_ENV}/bin/activate"
pip show pandas
STATUS=$?

if [ ${STATUS} -ne 0 ]
then
    pip install -r ${SCRIPT_DIR}/pydevs.txt
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PYTHON_ENV}/lib/python3.9/site-packages/tensorflow-plugins"
export PYTHONPATH="${PYTHONPATH}:${FOURASTRO_MODULE_PATH}"

