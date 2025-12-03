# Use official Python image
FROM python:3.10-slim

# Install system dependencies required by TensorFlow, OpenCV, DeepFace
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libglib2.0-dev \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages (TensorFlow must be installed separately)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Render port
EXPOSE 10000

# Start the Flask app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
