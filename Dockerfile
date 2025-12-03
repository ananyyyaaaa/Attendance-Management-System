# Use Python 3.10
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install OS dependencies required for OpenCV, DeepFace, and TensorFlow
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        git \
        wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip/setuptools/wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Use Gunicorn with 2 workers (for faster requests) and timeout 120s
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
