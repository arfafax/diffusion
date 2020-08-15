import os
import pickle
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
path = "./"
files = [ f for f in listdir(path) if (isfile(join(path, f)) and ".pkl" in f)]


with open(pkl, 'rb') as file:
    stuff = pickle.load( open( files[0], "rb" ) )
    print(len(stuff['progressive_samples']))
    print(len(stuff['progressive_samples'][0]))
    bkgr = Image.new('RGB', (256 * len(stuff['progressive_samples'][0]), 256 * len(stuff['progressive_samples'])))
    for idx,frame in enumerate(reverse(stuff['progressive_samples'])):
        for idy,sample in enumerate(frame):
            samp = np.clip(sample, -1, 1)
            formatted = ((samp + 1 ) * 255 / 2 ).astype('uint8')
            img = Image.fromarray(np.clip(formatted, 0, 255))
            bkgr.paste(img, (256 * idx, 256 * idy, 256 * idx + 256, 256 * idy + 256))
    bkgr.save('progressive_samples.jpg')
