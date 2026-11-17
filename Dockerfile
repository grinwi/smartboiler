FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY README.md README.md
COPY VERSION VERSION

RUN pip install --no-cache-dir -r requirements.txt

COPY src/smartboiler/ /app/src/smartboiler/

RUN pip install --no-cache-dir .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "smartboiler.controller"]
