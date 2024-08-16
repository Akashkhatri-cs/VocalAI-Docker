# Use an official Python runtime as a parent image
FROM python:3.12.4-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies, git, and Java
RUN apt-get update && \
    apt-get install -y ffmpeg git default-jre build-essential libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define environment variable for NLTK data
ENV NLTK_DATA=/usr/src/app/nltk_data

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Python dependencies and download NLTK data
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader -d /usr/src/app/nltk_data punkt stopwords && \
    rm -rf /tmp/*

# Ensure bcrypt is installed
RUN pip install bcrypt && \
    rm -rf /tmp/*

# Download and install spaCy model
RUN python -m spacy download en_core_web_sm && \
    rm -rf /tmp/*

# Expose port 80
EXPOSE 80

# Run the application with optimized Gunicorn settings
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--log-level", "debug", "--bind", "0.0.0.0:80", "app:app"]
