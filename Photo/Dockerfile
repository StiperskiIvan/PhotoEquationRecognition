FROM python:3.9-slim-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip3 install opencv-python-headless
WORKDIR /Photo
ADD . /Photo
EXPOSE 5000
CMD ["python","app.py"]