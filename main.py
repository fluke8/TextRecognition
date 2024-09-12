import torch
import argparse
import os

from src.utils import *
from src.config import *
from src.model import CRNN
from src.download_model import download_model

def main(image_path):
    model = CRNN(NUM_CLASSES, CHANNELS)

    if not os.path.isfile(MODEL_PATH):
        download_model()

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=torch.device(DEVICE)).state_dict())
    model.to(DEVICE)
    model.eval()

    if os.path.isdir(image_path):
        image_paths = [os.path.join(image_path, filename) 
                        for filename in os.listdir(image_path) 
                        if filename.split('.')[-1] in ['png', 'jpeg', 'jpg']]
    else:
        image_paths = [image_path]

    for image_path in image_paths:
        img = read_image(image_path)
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(DEVICE)

        output = model(img)
        preds = torch.argmax(output, dim=2)

        decoded_sequence = decode_sequence_with_blank(preds[0], SYMBOLS, BLANK_SYMBOL)

        print(f"Изображение {image_path}, распознанный текст: {decoded_sequence}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image(s)')
    
    args = parser.parse_args()
    main(args.image_path)
