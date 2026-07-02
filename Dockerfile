# Use a slim Python image for efficiency
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies (SQLite is required for your databases)
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Set environment variable to ensure the bot points to the right DB folder
ENV DB_PATH=/app

# Expose the port your Flask app runs on
EXPOSE 5001

# Command to run your application
CMD ["python", "demo_app.py"]
