FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY src/ ./src/
COPY assets/ ./assets/
COPY app.py ./
COPY .env.example ./

# Expose Streamlit default port
EXPOSE 8502

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
 