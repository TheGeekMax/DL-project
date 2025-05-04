

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    # check if models are loaded
    return jsonify({"status": "ok"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    """
    data = request.get_json()
    # load model and make prediction (Maxime bouge toi le cul)
    return jsonify({"prediction": "not implemented"}), 200


@app.route('/change-model', methods=['POST'])
def change_model():
    """
    Change model endpoint
    """
    data = request.get_json()
    # change used model for predictions (Maxime bouge toi le cul)
    return jsonify({"status": "not implemented"}), 200


if __name__ == '__main__':
    app.run(debug=True)
