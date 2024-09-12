import cv2
from image_processing import transform

def read_image(path):
    img = cv2.imread(path, 0)
    
    if img is None:
        raise FileNotFoundError(f"Изображение не найдено по пути: {path}")
    
    return img

def decode_sequence_with_blank(encoded, symbols, blank_symbol):
    decoded = []
    for index, code in enumerate(encoded):
        if code == blank_symbol or (index>0 and code==encoded[index-1]):
            continue
        decoded.append(symbols[code])
    
    return ''.join(decoded)

def prepare_image(img):
    return transform(img)