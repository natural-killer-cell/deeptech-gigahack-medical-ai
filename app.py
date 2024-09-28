import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Загрузка сохранённой модели
model = keras.models.load_model('./models/knee_model.keras')

# Параметры загрузки изображения
image_size = (180, 180)

@app.route('/predict', methods=['POST'])
def predict():
    # Получение файла изображения из запроса
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Загружаем и подготавливаем изображение
    img = image.load_img(io.BytesIO(file.read()), target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Создание оси батча

    # Предсказание
    predictions = model.predict(img_array)
    score = predictions[0][0]

    result = {
        "Patologie": f"{(100 * (1 - score)):.2f}%",
        "Norma": f"{(100 * score):.2f}%"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 
