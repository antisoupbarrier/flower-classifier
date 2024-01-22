FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

## Used to get keras_image_helper to work properly
RUN apt-get update && apt-get install -y git
#RUN apt-get update && apt-get install -y python3-numpy

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install numpy Pillow
RUN pipenv install --system --deploy

COPY ["gateway.py", "proto.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]