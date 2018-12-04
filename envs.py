import cv2
import numpy as np
from PIL import Image

def prepro(frame):
     """ prepro 210x210x3 uint8 frame into 6400 (80x80) 1D float vector """
     im = Image.fromarray(frame)
     im = im.convert('L'); # convert to gray scale
     frame = np.asarray(im).copy() # copy to numpy array
     frame = frame[34:34 + 210, :210]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
     frame = cv2.resize(frame, (105, 105))
     frame = cv2.resize(frame, (42, 42))
    # frame = frame.mean(2, keepdims=True)
     frame = frame.astype(np.float32)
    #frame *= (1.0 / 255.0)
     frame = np.moveaxis(frame, -1, 0)
     
     return normalize(frame)

def normalize(observation):
     state_mean = 0
     state_std = 0
     alpha = 0.9999
     num_steps = 0
     
     num_steps += 1
     state_mean = state_mean * alpha + \
     observation.mean() * (1 - alpha)
     state_std = state_std * alpha + \
     observation.std() * (1 - alpha)
     unbiased_mean = state_mean / (1 - pow(alpha, num_steps))
     unbiased_std = state_std / (1 - pow(alpha, num_steps))
     
     return (observation - unbiased_mean) / (unbiased_std + 1e-8)