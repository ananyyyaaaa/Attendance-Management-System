# Use Python 3.10
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install OS dependencies required for OpenCV
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip/setuptools/wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Use Gunicorn with single worker (LBPH is lightweight)
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:5000"]
