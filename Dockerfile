# Use an official Python runtime as a parent image
FROM python:3.11.4-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the app
EXPOSE 8080

# Make sure the script is executable
RUN chmod +x /app/start.sh

# Run the command to start FastAPI
CMD ["/app/start.sh"]


# gcloud builds submit --tag gcr.io/mlchallenge-79768/ml-challenge
# gcloud run deploy --image gcr.io/mlchallenge-79768/ml-challenge --platform managed

