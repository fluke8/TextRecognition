import os
import torch

MODEL_URL = "https://fluke8-public-bucket.storage.yandexcloud.net/text-recognition/model.pth"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")  
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYMBOLS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?"\',.-='
BLANK_SYMBOL = len(SYMBOLS)
# NUM_CLASSES = len(SYMBOLS) + 1

WIDTH = 256
HEIGHT = 64
