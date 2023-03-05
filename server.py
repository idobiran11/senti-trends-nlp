
from model_runner.model_runner import e2e_handler
from utils.constants import SourceNames, ModelNames, PreprocessNames
from models.news_sentiment import infer
from flask import Flask, json, Response, request

app = Flask(__name__)


@app.route("/")
def hello():
    return Response(json.dumps({"message": "server is up!"}), status=200, mimetype='application/json')


@app.route("/predict", methods=['POST'])
def run_inference_pipline():
    try:
        req = request.get_json()
    except:
        return Response(json.dumps({"message": "invalid request, pleas send JSON"}), status=400, mimetype='application/json')
    if not req.get('object', None):
        return Response(json.dumps({"message": "object name is required"}), status=400, mimetype='application/json')
    else:
        print(f"Running inference pipeline for object: {req['object']}")

        e2e_handler(object_name=req['object'], left_news_vendor=SourceNames.CNN,
                    right_news_vendor=SourceNames.FOX, model=ModelNames.SENT_NORM_NLTK,
                    preprocess=PreprocessNames.WITHOUT)

        return Response(json.dumps({"message": "running"}), status=200, mimetype='application/json')


if __name__ == "__main__":
    # running server
    app.run(host='localhost', port=5000)
