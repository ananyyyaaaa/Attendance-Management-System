FROM python:3.10-bullseye

WORKDIR /app

# Install OS dependencies
RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip/setuptools/wheel and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
