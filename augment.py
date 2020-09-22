from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

random.seed(0)

class BasicPolicy(object):
    def __init__(self, mirror_ratio = 0, flip_ratio = 0, color_change_ratio = 0, is_full_set_colors = False, add_noise_peak = 0.0, erase_ratio = -1.0):
        # Random color channel order
        from itertools import product, permutations
        self.indices = list(product([0,1,2], repeat = 3)) if is_full_set_colors else list(permutations(range(3), 3))
        self.indices.insert(0, [0,1,2]) # R,G,B
        self.add_noise_peak = add_noise_peak

        # Mirror and flip
        self.color_change_ratio = color_change_ratio
        self.mirror_ratio = mirror_ratio
        self.flip_ratio = flip_ratio

        # Erase
        self.erase_ratio = erase_ratio

    def __call__(self, img, depth):

        # 0) Add poisson noise (e.g. choose peak value 20)
        if self.add_noise_peak > 0:
            PEAK = self.add_noise_peak
            img = np.random.poisson(np.clip(img, 0, 1) * PEAK) / PEAK

        # 1) Color change
        policy_idx = random.randint(0, len(self.indices) - 1)
        if random.uniform(0, 1) >= self.color_change_ratio:
            policy_idx = 0

        img = img[...,list(self.indices[policy_idx])]

        # 2) Mirror image
        if random.uniform(0, 1) <= self.mirror_ratio:
            img = img[...,::-1,:]
            depth = depth[...,::-1,:]

        # 3) Flip image vertically
        if random.uniform(0, 1) < self.flip_ratio:
            img = img[...,::-1,:,:]
            depth = depth[...,::-1,:,:]

        # 4) Erase random box
        if random.uniform(0, 1) < self.erase_ratio:
            img = self.eraser(img)

        return img, depth




class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }


        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            #"rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
			"rotate": lambda img, magnitude: img,
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img