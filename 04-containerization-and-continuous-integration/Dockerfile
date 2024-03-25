# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install the required system libraries
RUN apt-get update && apt-get install -y libgomp1

# Copy the application code, model and requirements.txt into the container
COPY src/app.py src/score.py ./
COPY model/lightgbm_model.pkl ./model/
COPY src/requirements.txt ./

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app will run on
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "app.py"]