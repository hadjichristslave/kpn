from flask import Flask
from nltk.stem import PorterStemmer
from flask import request, Response
from sentiment_analyzer import *
import json

app = Flask(__name__)

tf_idf_vectorizer = Helpers.load_model(file_path=FILE_PATH, model_name=TF_IDF_VECTORIZER_NAME)
sentiment_predictor = Helpers.load_model(file_path=FILE_PATH, model_name=SENTIMENT_ANALYZER_NAME)
stemmer = PorterStemmer()


@app.route("/", methods=['GET'])
def index():
    return "Hello index"

@app.route("/predict_sentiment", methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    title = data.get('title', None)
    text = data.get('text', None)

    if title is None or text is None:
        return Response(response=json.dumps({}), status=204)


    request_data = title + ' ' + text

    if request_data == '':
        return Response(response=json.dumps({}), status=204)
    processed_text = Helpers.normalize_text(request_data, stemmer=stemmer)
    text_vectors = tf_idf_vectorizer.transform([processed_text])
    prediction = sentiment_predictor.predict(text_vectors)
    return Response(response=json.dumps(int(prediction[0])), status=200)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
