# Use an official lightweight Python image.
FROM python:3.9.6-slim

# Set the working directory in the container.
WORKDIR /app

# Install required system packages: poppler-utils (for pdf2image),
# libgl1-mesa-glx (for OpenCV), and libglib2.0-0 (for libgthread-2.0.so.0).
RUN apt-get update && \
    apt-get install -y poppler-utils libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variable to disable parallelism in tokenizers.
ENV TOKENIZERS_PARALLELISM=false

# Copy the requirements file into the container.
COPY requirements.txt requirements.txt

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that Render (or your local environment) will use.
EXPOSE 5000

# Run the application using Gunicorn, allowing variable substitution for PORT
# and increasing the worker timeout to 300 seconds.
CMD sh -c "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --timeout 300"
