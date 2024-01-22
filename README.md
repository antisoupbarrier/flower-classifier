## Problem description:

This project covers the training and deployment of an inference service for predicting flower types from an image.

The dataset used in this project is a collection of images comprising of 16 different flower species, with roughly 1000 images per flower.

You can find the dataset here: https://www.kaggle.com/datasets/l3llff/flowers

### Dataset Setup

You can find the dataset here: https://www.kaggle.com/datasets/l3llff/flowers

1. Download the files as .zip from Kaggle.

2. Unzip the contents of the zip into the repository main directory.

3. Provided you have tensorflow installed, the data is now ready for training.


## Pre-Trained Model setup

In order to save space on github, the model used by the project needs to be downloaded and prepared prior to configuring the docker images.

### Tensorflow Installed - Model Convert (Download model .h5)

Download the densenet .h5 model from this google drive: https://drive.google.com/file/d/19PIrN1XKsAAVWFe7iHAhu7zWQQwbkp0I/view?usp=sharing

Make sure the model 'densenet201_v3_15_0.913.h5' is saved to the main repo directory.

Run the model-convert.py script to convert the .h5 model to Tensorflow saved model format (for use by docker containers)

### Tensorflow Not Installed - Pre Converted (Download model pre-converted to saved model format)

Download the folder from this google drive: https://drive.google.com/drive/folders/1FMpiaZUMZXG81Kb3Xz8605JoUMDVNKbH?usp=sharing

Make sure the folder 'flower-model' and its contents are saved the main repo directory.

## Files

notebook.ipynb: Jupyter notebook used for EDA and model training. Additional models can be trained with this notebook, but it is recommened to use the provided model due to resources needed train models (high memory usage).

gateway.py: File for gateway service. Handles image preprocessing and prediction requests.

gateway-test.ipynb: Notebook to test gateway service in local environment (tensorflow must be configured to work properly).

proto.py: Script to handle tensorflow/protobuf conversion calculations for model prediction.

flower-model: Files for converted tensorflow model, in SavedModel format.

test.py: Simple test script for prediction. Provides a link to an image of an iris to the model, returning a response of flower class predictions.

train.py: Exported script to train a DenseNet121 model.

model-test.ipynb: Jupyter notebook to test model predictions for easy comparison of predicted and real flower types.

convert-model.ipynb: Notebook for loading in .h5 model and convert to .tflite or Keras Saved Model format.

image-model.dockerfile: Dockerfile for model service.

image-gateway.dockerfile: Dockerfile for gateway service.

docker-compose.yaml: Docker compose file for launching model and gateway services.

## Models Trained:

The models in this project were trained using tensorflow keras.

### Xception

Many Xception models were trained early on in the project - many of the training histories are saved within the main notebook file. These models and the associated accuracy/loss graphs may provide insight on the effects of modifying various parameters of keras models. Most of these attempts at training models do not converge.

### DenseNet

#### DenseNet121
#### DenseNet169
#### DenseNet201
The DenseNet models exhibited the highest accuracy and lowest losses of the keras applications used. Generally, DenseNet201 had the best accuracy, of the three versions, but with the caveat of being the most resource intensive to train and predict with. DenseNet121 had lower accuracy than DenseNet169, but DenseNet121 was the most efficient to train and predict with.

### InceptionV3


## Instructions to Run

### Training Models

If your computer is configured for training tensorflow keras models, the main notebook script contains the necessary code to train new models using multiple keras applications. The train.py script is a streamlined script to train a model with less options to modify. Please note that the image_size and batch_size used to train models on my system may need to be modified for other systems due to available memory resources.

### Running with Docker

In order to run the prediction service on a variety of systems, it is recommended to run using docker containers. For the following docker commands listed below, you must have your terminal window open to the main project directory containing the dockerfiles.

#### Build and Run model service

1. Run this command to build the docker image for the model service.
```bash
docker build -t flower-model:v1 -f image-model.dockerfile .
```

2. Run this command to make sure the docker image built properly. Close out with Ctrl-C before proceeding.
```bash
docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/flower-model:/models/flower-model/1" \
    -e MODEL_NAME="flower-model" \
    tensorflow/serving:2.7.0
```

#### Build and run gateway service

3. Run this command to build the docker image for the gateway service.
```bash
docker build -t flower-gateway:v1 -f image-gateway.dockerfile .
```

4. Run this command to make sure the docker image built properly. Close out with Ctrl-C
```bash
docker run -it --rm -p 9696:9696 flower-gateway:v1
```

#### Launch both services with Docker Compose

5. After sucessfully building the docker images for the model service and gateway service, you can run docker compose up to launch the two containers concurrently.

```bash
docker compose up
```

Note: If the docker containers are not launched in this way, they will not be able to communicate with eachother and the service will not work.

### Testing Predictions

After sucessfully launching the two docker containers with docker compose, you can now test the service locally. In a different terminal window, navigate to the project directory and run the following:

```bash
python test.py
```

The test.py script provides the prediction service with a url to an image of an iris. A functioning response will appear as such:

{'astilbe': 1.5793405054864706e-06, 'bellflower': 0.0009966525249183178, 'black_eyed_susan': 1.3982615065799564e-08, 'calendula': 3.125922376057133e-06, 'california_poppy': 5.542323378904257e-06, 'carnation': 7.318653842958156e-07, 'common_daisy': 2.6688809384722845e-07, 'coreopsis': 7.071327967622665e-09, 'daffodil': 7.023080570434104e-07, 'dandelion': 3.195454212345794e-07, 'iris': 0.9989344477653503, 'magnolia': 7.022159934422234e-06, 'rose': 4.4079670260543935e-06, 'sunflower': 3.9970956322576967e-07, 'tulip': 2.4865488512659795e-07, 'water_lily': 4.448892650543712e-05}

Feel free to update the url within the test.py script.

### Shutting Down 

When the prediction service is no longer needed, run the following to shut down all the docker containers launched by docker compose:

```bash
docker compose down
```

## Kubernetes Local Cluster

Note: The following requries kind and kubectl to be installed to deploy a local kubernetes cluster.

1. Create local cluster
```bash
kind create cluster
```
2. Load model docker image
```bash
kind load docker-image flower-model:v1
```
3. Apply model-deployment.yaml
```bash
kubectl apply -f model-deployment.yaml
```

4. Test model deployment - run get pod first and replace the name in the port forward command with the appropriate name.
```bash
kubectl get pod
kubectl port-forward tf-serving-model-name-from-get-pod 8500:8500
```

5. Setup Model Service
```bash
kubectl apply -f model-service.yaml
```
6. Test model service.
```bash
kubectl port-forward service/tf-serving-flower-model 8500:8500
```

7. Load gateway image
```bash
kind load docker-image flower-gateway:v1
```

8. Apply gateway deployment
```bash
kubectl apply -f gateway-deployment.yaml
```

9. Test gateway deployment.
```bash
kubectl get pod
kubectl port-forward gateway-pod-name-taken-from-get-pod 9696:9696
```

10. Apply gateway service.
```bash
kubectl apply -f gateway-service.yaml
```

11.  Test gateway service - make sure it runs sucessfully, Ctrl-C to close after.
```bash
kubectl port-forward service/gateway 8080:80
```

12. Test the model:
```bash
python test.py
```
