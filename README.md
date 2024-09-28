# DeepTechGigaHack Image Classification Project

This project is focused on classifying medical images into two categories: "Norma" (normal) and "Patologie" (pathological). It uses a convolutional neural network (CNN) model built with TensorFlow and Keras. The model is designed to identify whether an image falls into one of the two categories based on training data provided.

## Project Structure

- `train_and_save_model.py`: This script trains a convolutional neural network (CNN) on the dataset and saves the trained model.
- `predict_with_model.py`: This script loads the saved model and uses it to classify new images.
- `dataset/`: This folder should contain two subfolders: `Norma/` for normal images and `Patologie/` for pathological images.

## Prerequisites

To run this project, you'll need to have the following installed:

- Python 3.x
- TensorFlow
- Matplotlib (for visualizing data)
- NumPy (for array manipulation)
- Flask
- flask-cors

You can install the required libraries using the following command:

```bash
pip install tensorflow matplotlib Flask flask-cors
```

## How to Run

### 1. Training the Model

To train the model and save it, run the following command:

```bash
python train_and_save_model.py
```

### 2. Making prediction

After training the model, you can make predictions on new images. To do this, run:

```bash
python app.py
```

Make sure to update the image path in the script to point to the image you want to classify.
