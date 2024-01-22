FROM tensorflow/serving:2.7.0

COPY flower-model /models/flower-model/1
ENV MODEL_NAME="flower-model"