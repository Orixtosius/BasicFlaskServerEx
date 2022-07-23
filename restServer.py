from flask import Flask, request, jsonify
from model import modelResults
import time

app = Flask(__name__)

@app.route('/process', methods = ["POST"])
def process():
    content = request.json
    data = content["data"]
    print(f'Gotten datas are l1 {data[0]} and l2 {data[1]}')
    startTime = time.time()
    res = modelResults(int(data[0]), int(data[1]))
    lastLoss = res.history['loss'][-1]
    lastTestLoss = res.history['val_loss'][-1]

    passedTime = time.time() - startTime
    result = {"LastTrainLoss": lastLoss,
            "LastTestLoss": lastTestLoss,  
            "passedTime": passedTime}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)