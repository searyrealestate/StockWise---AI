# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Update pip and install the 'wheel' package before other requirements.
# This helps prevent 'metadata-generation-failed' errors.
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy the dependencies file first for efficient caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# The command to run your application when the container starts
CMD ["streamlit", "run", "stockwise_simulation.py", "--server.port=8501", "--server.address=0.0.0.0"]