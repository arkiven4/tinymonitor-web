@echo off
cd /d "%~dp0"
python manage.py collectstatic --noinput
@REM python manage.py runserver 0.0.0.0:8000
waitress-serve --listen=0.0.0.0:8000 web.wsgi:application