# COMS4732: Project 1 starter Python code
# Taken from: CS180 at UC Berkeley

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
imname = 'cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def shift_img(img, dy, dx):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def align(im1, im2):
    shift = 15
    best_score = float("inf") 
    best_shift = (0, 0)

    for i in range(-shift, shift):
        for j in range(-shift, shift):
            shifted_im1 = shift_img(im1, i, j)
            score = np.sum((im2 - shifted_im1) ** 2)
            if score < best_score:
                best_score = score
                best_shift = (i, j)

    return shift_img(im1, best_shift[0], best_shift[1])

ag = align(g, b)
ar = align(r, b)
# create a color image
im_out = np.dstack([ar, ag, b])

# save the image
fname = 'aligned_cathedral.jpg'
# Convert float (0-1) to uint8 (0-255) for saving
im_out_uint8 = (np.clip(im_out, 0, 1) * 255).astype(np.uint8)
skio.imsave(fname, im_out_uint8)


# display the image
skio.imshow(im_out)
skio.show()