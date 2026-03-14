import os
import pickle
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -------------------------------
# Load stacking ensemble components
# -------------------------------
print("Loading base models...")
resnet = tf.keras.models.load_model(os.path.join(MODEL_DIR, "Final_ResNet50.keras"), compile=False)
inception = tf.keras.models.load_model(os.path.join(MODEL_DIR, "Final_InceptionV3.keras"), compile=False)
xception = tf.keras.models.load_model(os.path.join(MODEL_DIR, "Final_Xception.keras"), compile=False)

print("Loading meta‑classifier...")
with open(os.path.join(MODEL_DIR, "meta_classifier.pkl"), "rb") as f:
    meta = pickle.load(f)

CLASS_NAMES = [
    'apple', 'bael', 'banana', 'black_plum', 'coconut', 'corn',
    'custard_apple', 'dragon_fruit', 'gooseberry', 'grape', 'guava',
    'hog_plum', 'jackfruit', 'longan', 'lychee', 'mandarin_orange',
    'mango', 'monkey_jack', 'muskmelon', 'natal_plum',
    'palmyra_palm', 'passion_fruit', 'pineapple', 'pomegranate',
    'pomelo', 'sapodilla', 'star_fruit', 'sweet_orange',
    'tarmind', 'water_apple', 'watermelon', 'wood_apple'
]

def load_and_preprocess(image_path, target_size=(224, 224)):
    """Exactly as in Colab: PIL open, resize, convert to float32 numpy array (0-255)."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)   # shape (h,w,3), values 0..255
    return img_array

def predict_stacking(image_path):
    # Load image at 224x224 (ResNet50 size) – same as Colab
    img_224 = load_and_preprocess(image_path, (224, 224))
    img_224_batch = np.expand_dims(img_224, axis=0)   # (1,224,224,3)

    # Resize for Inception (299x299) and Xception (256x256)
    img_299 = tf.image.resize(img_224_batch, (299, 299)).numpy()
    img_256 = tf.image.resize(img_224_batch, (256, 256)).numpy()

    # Get predictions (probabilities) from each base model
    pred_r = resnet.predict(img_224_batch, verbose=0)   # (1,32)
    pred_i = inception.predict(img_299, verbose=0)      # (1,32)
    pred_x = xception.predict(img_256, verbose=0)       # (1,32)

    # Stack features for meta‑classifier
    features = np.column_stack([pred_r, pred_i, pred_x])  # (1,96)

    # Meta‑classifier prediction
    class_idx = meta.predict(features)[0]
    proba = meta.predict_proba(features)[0]
    confidence = float(proba[class_idx]) * 100

    # (Optional) Print debug info – uncomment if needed
    # print("Top 3 predictions from each base model:")
    # top_r = np.argsort(pred_r[0])[::-1][:3]
    # top_i = np.argsort(pred_i[0])[::-1][:3]
    # top_x = np.argsort(pred_x[0])[::-1][:3]
    # print("ResNet:", [(CLASS_NAMES[i], pred_r[0][i]) for i in top_r])
    # print("Inception:", [(CLASS_NAMES[i], pred_i[0][i]) for i in top_i])
    # print("Xception:", [(CLASS_NAMES[i], pred_x[0][i]) for i in top_x])
    # print("Meta probabilities top5:", [(CLASS_NAMES[i], proba[i]) for i in np.argsort(proba)[::-1][:5]])

    return CLASS_NAMES[class_idx], confidence

def home(request):
    context = {}
    if request.method == "POST" and request.FILES.get("image"):
        img_file = request.FILES["image"]
        fs = FileSystemStorage()
        file_path = fs.save(img_file.name, img_file)
        full_path = fs.path(file_path)

        try:
            pred_class, confidence = predict_stacking(full_path)
            context = {
                "prediction": pred_class,
                "confidence": f"{confidence:.2f}%",
                "image_url": fs.url(file_path),
            }
        except Exception as e:
            context = {"error": str(e)}

    return render(request, "home.html", context)

def model_result_page(request):
    return render(request, "modelResult.html")

def fruits_view(request):
    fruits = [
        {"name": "Apple", "image": "images/apple.jpg"},
        {"name": "Bael", "image": "images/bael.jpg"},
        {"name": "Banana", "image": "images/banana.jpg"},
        {"name": "Black Plum", "image": "images/black_plum.jpg"},
        {"name": "Coconut", "image": "images/coconut.jpg"},
        {"name": "Corn", "image": "images/corn.jpg"},
        {"name": "Custard Apple", "image": "images/custard_apple.jpg"},
        {"name": "Dragon Fruit", "image": "images/dragon_fruit.jpg"},
        {"name": "Gooseberry", "image": "images/gooseberry.jpg"},
        {"name": "Grape", "image": "images/grape.jpg"},
        {"name": "Guava", "image": "images/guava.jpg"},
        {"name": "Hog Plum", "image": "images/hog_plum.jpg"},
        {"name": "Jackfruit", "image": "images/jackfruit.jpg"},
        {"name": "Longan", "image": "images/longan.jpg"},
        {"name": "Lychee", "image": "images/lychee.jpg"},
        {"name": "Mandarin Orange", "image": "images/mandarin_orange.jpg"},
        {"name": "Mango", "image": "images/mango.jpg"},
        {"name": "Monkey Jack", "image": "images/monkey_jack.jpg"},
        {"name": "Muskmelon", "image": "images/muskmelon.jpeg"},
        {"name": "Natal Plum", "image": "images/natal_plum.jpg"},
        {"name": "Palmyra Palm", "image": "images/palmyra_palm.jpg"},
        {"name": "Passion Fruit", "image": "images/passion_fruit.jpg"},
        {"name": "Pineapple", "image": "images/pineapple.jpg"},
        {"name": "Pomegranate", "image": "images/pomegranate.jpg"},
        {"name": "Pomelo", "image": "images/pomelo.jpg"},
        {"name": "Sapodilla", "image": "images/sapodilla.jpg"},
        {"name": "Star Fruit", "image": "images/star_fruit.jpg"},
        {"name": "Sweet Orange", "image": "images/sweet_orange.jpg"},
        {"name": "Tamarind", "image": "images/tarmind.jpg"},
        {"name": "Water Apple", "image": "images/water_apple.jpg"},
        {"name": "Watermelon", "image": "images/watermelon.jpg"},
        {"name": "Wood Apple", "image": "images/wood_apple.jpg"},
    ]

    return render(request, "fruits.html", {"fruits": fruits})