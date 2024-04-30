# Use an official OpenSim with Python runtime as a parent image
FROM stanfordnmbl/opensim-python:latest

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies for general operation
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . /app

# Set timezone to Pakistan
ENV TZ=Asia/Karachi
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=joint_angles.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

