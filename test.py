import requests

url = 'http://localhost:9696/predict'

# URL for the image to be predicted
data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Purple-Iris_pn.jpg/1200px-Purple-Iris_pn.jpg'}

result = requests.post(url, json=data).json()
print(result)