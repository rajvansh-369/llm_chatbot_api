# Use official CUDA-enabled PyTorch image for GPU support
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if exists, otherwise install in Dockerfile
COPY requirements.txt .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask default port
EXPOSE 5000

# Run app
CMD ["python", "api.py"]
