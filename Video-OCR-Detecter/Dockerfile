FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# Ensure the package lists directory exists
RUN mkdir -p /var/lib/apt/lists/partial

# Verify Tesseract-OCR installation
RUN tesseract --version

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Start the application
CMD ["gunicorn", "app:app"]
