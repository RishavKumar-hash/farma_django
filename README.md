
# Farma - Leaf Disease Detection Web App 🌿

**Farma** is a Django-based web application that allows farmers to upload images of plant leaves and uses a Machine Learning model to detect if the leaf is healthy or diseased.

---

## ✅ Features

- Upload and preview leaf images
- Predicts disease using a CNN model (MobileNetV2-based)
- Result shown with class label and confidence score
- Custom Django Admin for managing records
- Production-ready setup (Docker, Gunicorn, etc.)

---

## 🧠 ML Model (Real)

This app uses a **MobileNetV2-based CNN model**, trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

### 🪄 Included Model
We included a placeholder `model.h5` file (~50MB). To replace it with a **real trained model**, follow these steps:

### 🔁 Replace Placeholder with Real Model

1. **Download Pre-trained Model** (Option 1: Pretrained)
   - Use this community-trained version: [Leaf Disease Model - Kaggle](https://www.kaggle.com/code/smaranjitghose/plant-disease-detection-tf-hub)
   - Export the trained `.h5` file after training
   - Place it in `leafscanner/ml_model/model.h5`

2. **Train Your Own Model** (Option 2)
   - Dataset: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
   - Python code sample to train:

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(4, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_gen.flow_from_directory("path_to_dataset", target_size=(224, 224), class_mode="categorical", subset="training")
val_data = train_gen.flow_from_directory("path_to_dataset", target_size=(224, 224), class_mode="categorical", subset="validation")

model.fit(train_data, validation_data=val_data, epochs=10)
model.save("model.h5")
```

3. **Put the model** in:
```
farma_django/leafscanner/ml_model/model.h5
```

---

## 🚀 Deployment

### Local (Docker Compose)
```bash
docker-compose up --build
```

### Heroku Deployment
- Add `Procfile`, `gunicorn`, and follow Heroku Python deployment guide

### AWS EC2 / Elastic Beanstalk
- Use the Dockerfile or a Gunicorn + Nginx production stack

---

## 🔐 Admin Panel
- Visit `/admin/` to manage records
- Superuser setup: `python manage.py createsuperuser`

---

## 📦 Requirements (requirements.txt)

- Django>=3.2
- tensorflow
- pillow
- gunicorn
- python-dotenv

---

## 📝 License

This project is licensed under the MIT License.
