#!/bin/bash

set -e

# activate our virtual environment here
#. /opt/pysetup/.venv/bin/activate

WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-DEBUG}

gunicorn --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8079 -w $WORKERS wsgi:app --log-level $LOG_LEVEL
