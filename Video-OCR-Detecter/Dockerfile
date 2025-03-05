FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libtesseract-dev

# Ensure the package lists directory exists
RUN mkdir -p /var/lib/apt/lists/partial

# Set the working directory
WORKDIR /app

# Copy the application code and Tesseract-OCR binaries
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Start the application
CMD ["gunicorn", "app:app"]