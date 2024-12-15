FROM python:3.10.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install cython

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["python3", "router.py"]