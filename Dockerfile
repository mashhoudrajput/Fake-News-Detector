FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Train model when building image
RUN python3 train.py

EXPOSE 5000
CMD ["python3", "app.py"]

