import tensorflow as tf
from tensorflow import keras

# Загрузка сохранённой модели
model = keras.models.load_model('./models/knee_model.keras')

# Параметры загрузки изображения
image_size = (180, 180)

# Инференс на новом изображении
img = keras.preprocessing.image.load_img(
    "/home/dezorel/dezorel-library/hackaton/DeepTechGigaHack/deeptech-gigahack-medical-ai/dataset/Patologie/5265188666182064881.jpg", 
    target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Создание оси батча

# Предсказание
predictions = model.predict(img_array)

score = predictions[0][0]
print(f"Это изображение {(100 * (1 - score)):.2f}% Patologie и {(100 * score):.2f}% Norma.")
