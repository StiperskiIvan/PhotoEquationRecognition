# init a base image (Alpine is small Linux distro)
FROM python:3.9-slim-buster
# Step 2: Add requirements.txt file
COPY requirements.txt requirements.txt
# Step 3:  Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt
RUN pip3 install opencv-python-headless
# define the present working directory
WORKDIR /Photo
# copy the contents into the working dir
ADD . /Photo
# Step 6: Expose the port Flask is running on
EXPOSE 5000
# define the command to start the container
CMD ["python","app.py"]