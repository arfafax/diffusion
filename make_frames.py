import pickle

from os import listdir
from os.path import isfile, join
path = "./"
files = [ f for f in listdir(path) if (isfile(join(path, f)) and ".pkl" in f)]
samples = pickle.load( open( files[0], "rb" ) )

from PIL import Image
import numpy as np

print(len(samples['progressive_samples']))
print(len(samples['progressive_samples'][0]))
for index,frames in enumerate(samples['progressive_samples']):
    for frame,sample in enumerate(frames):
        samp = np.clip(sample, -1, 1)
        formatted = ((samp + 1) * 255 / 2).astype('uint8')
    

        img = Image.fromarray(np.clip(formatted,0,255))
        #display(img)
        img.save("frames/foo-{:04d}.png".format(index*len(samples['progressive_samples'][0])+frame))

