#!/bin/sh
exec gunicorn -k uvicorn.workers.UvicornWorker -b :5000 --timeout 6000 --max-requests=5000 --max-requests-jitter=250 api:app