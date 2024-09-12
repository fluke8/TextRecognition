import os
import requests
from .config import MODEL_PATH, MODEL_DIR, MODEL_URL

def download_model():
    response = requests.get(MODEL_URL)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as file:
            file.write(response.content)
        print("Модель успешно скачана и сохранена в:", MODEL_PATH)
    else:
        print("Ошибка при скачивании модели:", response.status_code)

