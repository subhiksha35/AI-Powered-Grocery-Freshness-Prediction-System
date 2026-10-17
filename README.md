# AI-Powered Grocery Freshness Prediction System

A beginner-friendly academic project that demonstrates image classification using Python, OpenCV, TensorFlow/Keras, and Flask. Users upload an image and the trained model predicts one of the dataset's freshness classes.

> **Important:** This is an educational prototype. Its output is not a food-safety guarantee. Do not eat or discard food based only on this prediction.

## Features

- Upload a JPG, JPEG, PNG, or WEBP image through a web interface.
- Use OpenCV to read, resize, and prepare the image.
- Train a CNN image-classification model with TensorFlow/Keras.
- Show the predicted class and model confidence.
- Run the app locally with Flask.

## Technologies

- Python
- Flask
- OpenCV
- TensorFlow / Keras (CNN)
- HTML and CSS
- Google Colab can be used to train the model if a local computer does not have enough resources.

## Project structure

```text
AI_Powered_Grocery_Freshness_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/
│   ├── train/
│   │   ├── Fresh/
│   │   ├── Slightly_Spoiled/
│   │   └── Spoiled/
│   └── validation/
│       ├── Fresh/
│       ├── Slightly_Spoiled/
│       └── Spoiled/
├── model/
│   └── .gitkeep
├── static/
│   ├── style.css
│   └── uploads/
│       └── .gitkeep
└── templates/
    ├── index.html
    └── result.html
```

## 1. Prepare the dataset

Add correctly labelled fruit and vegetable images to the matching folders. The folder names are the class labels, so use exactly the same folder names for both training and validation.

Example:

```text
dataset/train/Fresh/apple_001.jpg
dataset/train/Slightly_Spoiled/apple_021.jpg
dataset/train/Spoiled/apple_041.jpg
dataset/validation/Fresh/apple_101.jpg
dataset/validation/Slightly_Spoiled/apple_121.jpg
dataset/validation/Spoiled/apple_141.jpg
```

Use a reliable, appropriately licensed dataset. Keep separate images for training and validation; do not put duplicate images in both sets. Aim for a balanced number of images in each class. The folders in this starter project are empty, so the model cannot be trained until you add images.

## 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

If TensorFlow installation fails for your Python version or operating system, use a supported Python environment or Google Colab.

## 3. Train the model

From the project root:

```bash
python train_model.py
```

The script trains a basic CNN and creates:

- `model/freshness_model.keras`
- `model/class_names.json`

Training time and accuracy depend on the dataset, hardware, and image quality. Do not claim a particular accuracy until you have trained and evaluated the model.

## 4. Run the web app

After training completes:

```bash
python app.py
```

Open the local URL shown in the terminal (usually `http://127.0.0.1:5000`) and upload an image.

## 5. Using Google Colab

You can run the training script in a Colab environment after uploading the project and dataset. Make sure the `dataset/train` and `dataset/validation` directories are available at the expected project path. Download the generated `freshness_model.keras` and `class_names.json` into the local project's `model/` folder before running Flask locally.

## Limitations

- The starter repository does not include a dataset or a trained model.
- A CNN predicts patterns learned from its training data; poor or unrepresentative data can produce unreliable results.
- Image appearance alone cannot reliably establish whether food is safe to eat.
- This project is a prototype and requires testing before any real-world use.

## Resume description (use only after implementing and testing)

**AI-Powered Grocery Freshness Prediction System**  
*Python, OpenCV, TensorFlow/Keras (CNN), Flask, HTML, CSS*

- Built a Flask web application prototype for image-based grocery freshness classification.
- Used OpenCV to preprocess uploaded images and a CNN model to classify images into freshness categories.
- Trained and evaluated the model using labelled training and validation images.

Replace or adjust these points to match the features and results you actually implement.
