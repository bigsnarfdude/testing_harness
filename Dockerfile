# Use an official Python 3.9 slim base image
FROM python:3.9-slim

# Set up a working directory
WORKDIR /app

# Copy the requirements.txt file into the image
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scripts/async_harness.py file into the image
COPY scripts/async_harness.py scripts/

# Copy the benchmark_data directory into the image
COPY benchmark_data/ benchmark_data/

# Define environment variables for default values
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV MODEL_NAME=llama3.2:latest
ENV OUTPUT_DIR=/app/results
ENV VERBOSE_LOGGING=false

# Set an entrypoint that allows running async_harness.py
ENTRYPOINT ["python", "scripts/async_harness.py"]

# Default command to show the help message
CMD ["--help"]
