# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt requirements.txt

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that Render will use.
EXPOSE 10000

# Run the application using Gunicorn.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
