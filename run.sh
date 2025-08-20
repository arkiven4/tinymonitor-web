#!/bin/bash
cd "$(dirname "$0")"
python3.9 manage.py collectstatic --noinput
python3.9 manage.py runserver 0.0.0.0:8000

