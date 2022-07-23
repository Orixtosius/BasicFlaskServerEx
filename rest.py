import requests
import json

l1 = int(input('Please specify neuron in first hidden layer....\n'))
l2 = int(input('Please specify neuron in second hidden layer....\n'))
response = requests.post('http://localhost:6000/process',
                         headers={'Content-Type': 'application/json'},
                         data=json.dumps({'data': (l1, l2)}))

print(response.json())