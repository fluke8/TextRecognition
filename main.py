import torch
import argparse
import os

from src.utils import *
from src.config import *

def main(image_path):
    model = torch.load(MODEL_PATH)

    print(f"Image directory: {image_path}")

    print(os.listdir("/"))
    print(os.listdir(image_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image(s)')
    
    args = parser.parse_args()
    main(args.image_path)
