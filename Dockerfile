FROM python:3.10-slim

WORKDIR /home

COPY requirements.txt /home

RUN pip install --no-cache-dir -r requirements.txt

COPY . .