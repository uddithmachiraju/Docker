# Use the python as base image
FROM python:3.10

WORKDIR /app 

# Copy project files into container 
COPY requirements.txt . 
COPY app.py .
COPY model/config.py . 
COPY model/interface.py . 
COPY model/main.py . 
COPY model/net.py .
COPY model/test.py .
COPY model/train.py . 

# Install dependencies
RUN pip install -r requirements.txt 

# Expose the port for Flask
EXPOSE 5000

# Run Flask app 
CMD ["python", "app.py"] 