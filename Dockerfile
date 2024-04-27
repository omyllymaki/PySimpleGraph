FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    graphviz \
    xdg-utils

RUN python -m pip install tiny-dag

WORKDIR /app
COPY src/sample.py /app

CMD ["python", "sample.py"]