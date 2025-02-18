from flask import Flask, jsonify 
from model.interface import predict 

app = Flask(__name__) 

@app.route('/') 
def home():
    return "MNIST Digit Recogination API is Running..." 

@app.route('/predict', methods = ['GET']) 
def predict_digit():
    image_path = 'test_image.jpg' 
    with open(image_path, "rb") as img_file:
        prediction = predict(img_file)

    return jsonify({'Prediction': prediction}) 

if __name__ == "__main__":
    app.run(debug = True) 