import torch
import torch.nn as nn
from torchvision.transforms import functional
from pytorch_wavelets import DWTForward, DWTInverse

import cv2
import numpy as np
from PIL import Image
import random

import torch
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

totensor = torchvision.transforms.ToTensor()
interpolation = functional.InterpolationMode('bilinear')

class GaussianBlurAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 3
        # self.sigma = 1.9
    
    def forward(self, image):
        blurred_img = functional.gaussian_blur(image, self.kernel_size)
        return blurred_img

class RotationAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = 30


    def forward(self, image):
        rand_num = random.uniform(-1, 1)
        rotated_img = functional.rotate(image, self.angle*rand_num)#, expand = True)
        return rotated_img

class CropAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 0.1 
    
    def forward(self, image):
        scale = np.sqrt(self.scale)
        edges_size = [int(s* scale) for s in image.size][::-1]
        cropped_img = functional.center_crop(image, edges_size)
        return cropped_img

class ResizeAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 0.7
    def forward(self, image):
        scale = np.sqrt(self.scale)
        edges_size = [int(s*scale) for s in image.size][::-1]
        resized_img = functional.resize(image, edges_size,interpolation=interpolation, antialias=True)
        return resized_img

class NoiseAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = 0.01

    def forward(self, image):
        tensor_img = totensor(image)
        noised_image = tensor_img + (self.sigma **2) * torch.randn_like(tensor_img)
        tensor_to_PIL = torchvision.transforms.ToPILImage()

        return tensor_to_PIL(noised_image)


class JPEGCompressAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.magnitude = 10

    def forward(self, image):
        # Convert the Pillow image to a NumPy array (OpenCV image)
        cv_img = np.array(image)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.magnitude]
        result, encimg = cv2.imencode('.jpg', cv_img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
        return decimg

class Attacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_layers = []
        self.noise_layers.append(GaussianBlurAttack())
        self.noise_layers.append(RotationAttack())
        self.noise_layers.append(CropAttack())
        self.noise_layers.append(ResizeAttack())
        self.noise_layers.append(NoiseAttack())
        self.noise_layers.append(JPEGCompressAttack())


    def forward(self, image, idx):
        # random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        random_noise_layer = self.noise_layers[idx]

        return random_noise_layer(image)