# import requests
# import pandas as pd
# import json

# inputs = [1, 2, 3, 4]
# inputs = json.dumps({'array': inputs})

# # Flask client call
# res = requests.post('http://0.0.0.0:5000/predict', json=inputs).json()
# print(res)

curl -X POST http://0.0.0.0:5000/predict -H "Content-Type: application/json" -d '{"array": [1, 2, 3, 4]}'