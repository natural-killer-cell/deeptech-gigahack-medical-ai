import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os

# Путь к локальным данным
DATADIR = '/home/dezorel/dezorel-library/hackaton/DeepTechGigaHack/deeptech-gigahack-medical-ai/dataset'  
CATEGORIES = ["Norma", "Patologie"]

# Удаление поврежденных изображений
num_skipped = 0
for category in CATEGORIES:
    folder_path = os.path.join(DATADIR, category)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print(f"Удалено {num_skipped} изображений")

# Параметры загрузки данных
image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATADIR,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATADIR,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Аугментация данных
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),  # Случайный сдвиг
        layers.RandomCrop(height=image_size[0], width=image_size[1], seed=1337),  # Случайный обрез
    ]
)


# Подготовка датасета для повышения производительности
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Используем предобученную модель
base_model = keras.applications.MobileNetV2(input_shape=image_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Замораживаем базовую модель

# Добавляем свои слои
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(CATEGORIES), activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=outputs)

# Компиляция модели
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Меньшая скорость обучения
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Тренировка модели
epochs = 20

class_weight = {0: 3.0, 1: 1.0} # Пример: увеличиваем вес для класса "Норма"

model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[early_stopping],
)

# Сохранение модели
model.save('./models/knee_model.keras')
print("Модель сохранена как knee_model.keras")
