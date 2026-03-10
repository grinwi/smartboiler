FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY README.md README.md
COPY VERSION VERSION

RUN pip install --no-cache-dir -r requirements.txt

COPY src/smartboiler/ /app/src/smartboiler/

# Substitute VERSION placeholder so setup.py install succeeds locally/in CI
RUN VERSION=$(cat VERSION | tr -d '[:space:]') \
    && sed -i "s/{{VERSION_PLACEHOLDER}}/${VERSION}/g" setup.py \
    && pip install --no-cache-dir .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "smartboiler.controller"]
