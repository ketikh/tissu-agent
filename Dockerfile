FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create directories
RUN mkdir -p data static/products static/uploads

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "server.py"]
