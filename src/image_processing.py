from torchvision import transforms
import torch
from config import WIDTH, HEIGHT


class ResizePadding:
    def __init__(self, target_size, mode='width', centered=False):
        self.target_size = target_size
        self.mode = mode 
        self.centered = centered

    def __call__(self, img):
        channels, height, width = img.shape
        
        if self.mode == 'width' and width < self.target_size:
            padding_width = self.target_size - width
            
            if self.centered:
                left_padding_width = right_padding_width = padding_width // 2
            else:
                left_padding_width = random.randint(0, padding_width)
                right_padding_width = padding_width - left_padding_width
            
            left_padding = torch.zeros(channels, height, left_padding_width)
            right_padding = torch.zeros(channels, height, right_padding_width)
            padded_img = torch.cat((left_padding, img, right_padding), dim=2)

        elif self.mode == 'height' and height < self.target_size:
            padding_height = self.target_size - height
            
            if self.centered:
                upper_padding_height = lower_padding_height = padding_height // 2
            else:
                upper_padding_height = random.randint(0, padding_height)
                lower_padding_height = padding_height - upper_padding_height
            
            upper_padding = torch.zeros(channels, upper_padding_height, width)
            lower_padding = torch.zeros(channels, lower_padding_height, width)
            padded_img = torch.cat((upper_padding, img, lower_padding), dim=1)

        else:
            padded_img = img

        return padded_img

    def __str__(self) -> str:
        return f"ResizePadding {self.target_size}, mode={self.mode}, centered={self.centered}"

    
class RotateTransform:
    def __init__(self, angle=-90, expand=True):
        self.angle = angle
        self.expand = expand

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle, expand=self.expand)

    def __str__(self) -> str:
        return (f"RotateTransform {self.angle, self.expand}")


transform = transforms.Compose([
                        transforms.ToTensor(),
                        ResizePadding(WIDTH, centered=True, mode='width'),
                        ResizePadding(HEIGHT, centered=True, mode='height'),
                        RotateTransform(angle=-90, expand=True),
                        transforms.Resize((WIDTH, HEIGHT))  
                    ])
