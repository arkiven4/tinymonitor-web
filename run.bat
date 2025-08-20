@echo off
cd /d "%~dp0"
python manage.py collectstatic --noinput
python manage.py runserver 0.0.0.0:8000