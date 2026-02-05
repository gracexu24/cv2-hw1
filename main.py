# COMS4732: Project 1 starter Python code
# Taken from: CS180 at UC Berkeley

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as skt
import json
# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def shift_img(img, dy, dx):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def crop(img, border_frac=0.08):
    h, w = img.shape
    bh = int(h * border_frac)
    bw = int(w * border_frac)
    return img[bh:h-bh, bw:w-bw]

# eclidean distance method 
def align(im1, im2):
    shift = 15
    best_score = float("inf") 
    best_shift = (0, 0)

    for i in range(-shift, shift):
        for j in range(-shift, shift):
            shifted_im1 = shift_img(im1, i, j)
            # u shld try different alignment methods please!
            score = np.sum((crop(im2) - crop(shifted_im1)) ** 2)
            if score < best_score:
                best_score = score
                best_shift = (i, j)
    return best_shift

# normal cross-correlation method
def align_test(im1, im2):
    shift = 15
    best_score = float("-inf") 
    best_shift = (0, 0)
    for i in range(-shift, shift):
        for j in range(-shift, shift):
            shifted_im1 = shift_img(im1, i, j)
            score = ncc(crop(im2), crop(shifted_im1)) 
            if score > best_score:
                best_score = score
                best_shift = (i, j)
    return best_shift

def ncc(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.sum(a * b) / np.sqrt(np.sum(a*a) * np.sum(b*b))

def downsample(img):
    return skt.rescale(img, 0.5, anti_aliasing=True, channel_axis=None)

def align_pyramid(im1, im2):
    h, w = im1.shape

    # ---- Base case: image small enough → brute force ----
    if min(h, w) < 400:
        return align(im1, im2)

    # ---- Recursive case: go to smaller images first ----
    im1_small = downsample(im1)
    im2_small = downsample(im2)

    # Get rough shift from smaller images
    dy_small, dx_small = align_pyramid(im1_small, im2_small)

    # Scale shift back to current resolution
    dy = dy_small * 2
    dx = dx_small * 2

    best_score = float("inf") 
    best_shift = (0, 0)

    # Refine search around the estimated shift
    for ddy in range(dy - 6, dy + 7):
        for ddx in range(dx - 6, dx + 7):
            shifted_im1 = shift_img(im1, ddy, ddx)
            im1_crop = crop(shifted_im1)
            im2_crop = crop(im2)
            score = np.sum((im2_crop - im1_crop) ** 2)

            if score < best_score:
                best_score = score
                best_shift = (ddy, ddx)

    return best_shift




def process_simple_image(imname):
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
    ag_best_shift = align(g, b)
    ag = shift_img(g, ag_best_shift[0], ag_best_shift[1])
    ar_best_shift = align(r, b)
    ar = shift_img(r, ar_best_shift[0], ar_best_shift[1])
    
    # create a color image
    im_out = np.dstack([ar, ag, b])
    
    # save the image
    if imname.endswith('.jpg'):
        output_name = imname.replace('.jpg', '_simple_aligned.jpg')
    elif imname.endswith('.tif'):
        output_name = imname.replace('.tif', '_simple_aligned.jpg')
    else:
        output_name = imname.rsplit('.', 1)[0] + '_simple_aligned.jpg'
    # Convert float (0-1) to uint8 (0-255) for saving
    im_out_uint8 = (np.clip(im_out, 0, 1) * 255).astype(np.uint8)
    skio.imsave(output_name, im_out_uint8)
    
    print(f"Processed {imname} -> {output_name}")
    
    # Return offsets
    return output_name, {
        'green': {'dy': int(ag_best_shift[0]), 'dx': int(ag_best_shift[1])},
        'red': {'dy': int(ar_best_shift[0]), 'dx': int(ar_best_shift[1])}
    }

def process_simple_image_ncc(imname):
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
    ag_best_shift = align_test(g, b)
    ag = shift_img(g, ag_best_shift[0], ag_best_shift[1])
    ar_best_shift = align_test(r, b)
    ar = shift_img(r, ar_best_shift[0], ar_best_shift[1])
    
    # create a color image
    im_out = np.dstack([ar, ag, b])
    
    # save the image
    if imname.endswith('.jpg'):
        output_name = imname.replace('.jpg', '_simple_ncc_aligned.jpg')
    elif imname.endswith('.tif'):
        output_name = imname.replace('.tif', '_simple_ncc_aligned.jpg')
    else:
        output_name = imname.rsplit('.', 1)[0] + '_simple_ncc_aligned.jpg'
    # Convert float (0-1) to uint8 (0-255) for saving
    im_out_uint8 = (np.clip(im_out, 0, 1) * 255).astype(np.uint8)
    skio.imsave(output_name, im_out_uint8)
    
    print(f"Processed {imname} -> {output_name}")
    
    # Return offsets
    return output_name, {
        'green': {'dy': int(ag_best_shift[0]), 'dx': int(ag_best_shift[1])},
        'red': {'dy': int(ar_best_shift[0]), 'dx': int(ar_best_shift[1])}
    }


def process_pyramid_image(imname):
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
    ag_best_shift = align_pyramid(g, b)
    ag = shift_img(g, ag_best_shift[0], ag_best_shift[1])
    ar_best_shift = align_pyramid(r, b)
    ar = shift_img(r, ar_best_shift[0], ar_best_shift[1])
    
    # create a color image
    im_out = np.dstack([ar, ag, b])
    
    # save the image
    if imname.endswith('.jpg'):
        output_name = imname.replace('.jpg', '_pyramid_aligned.jpg')
    elif imname.endswith('.tif'):
        output_name = imname.replace('.tif', '_pyramid_aligned.jpg')
    else:
        output_name = imname.rsplit('.', 1)[0] + '_pyramid_aligned.jpg'
    # Convert float (0-1) to uint8 (0-255) for saving
    im_out_uint8 = (np.clip(im_out, 0, 1) * 255).astype(np.uint8)
    skio.imsave(output_name, im_out_uint8)
    
    print(f"Processed {imname} -> {output_name}")
    
    # Return offsets
    return output_name, {
        'green': {'dy': int(ag_best_shift[0]), 'dx': int(ag_best_shift[1])},
        'red': {'dy': int(ar_best_shift[0]), 'dx': int(ar_best_shift[1])}
    }

# Process all simple images
simple_image_files = ['cathedral.jpg', 'tobolsk.jpg', 'monastery.jpg']
simple_output_files = []
simple_offsets = {}
simple_ncc_output_files = []
simple_ncc_offsets = {}
all_image_files = ['cathedral.jpg', 'tobolsk.jpg', 'monastery.jpg', 'melons.tif', 'church.tif', 'emir.tif', 'harvesters.tif', 'icon.tif','italil.tif','lastochikino.tif', 'lugano.tif', 'self_portrait.tif', 'siren.tif', 'three_generations.tif']
pyramid_output_files = []
pyramid_offsets = {}
own_image_files = ['ownimage1.tif', 'ownimage2.tif', 'ownimage3.tif']
own_output_files = []
own_offsets = {}

print("Processing images with simple alignment (Euclidean distance)...")
for imname in simple_image_files:
    try:
        output_name, offsets = process_simple_image(imname)
        simple_output_files.append(output_name)
        simple_offsets[imname] = offsets
    except Exception as e:
        print(f"Error processing {imname}: {e}")

print("\nProcessing images with simple alignment (Normalized Cross-Correlation)...")
for imname in simple_image_files:
    try:
        output_name, offsets = process_simple_image_ncc(imname)
        simple_ncc_output_files.append(output_name)
        simple_ncc_offsets[imname] = offsets
    except Exception as e:
        print(f"Error processing {imname}: {e}")


print("\nProcessing images with pyramid alignment...")
for imname in all_image_files:
    try:
        output_name, offsets = process_pyramid_image(imname)
        pyramid_output_files.append(output_name)
        pyramid_offsets[imname] = offsets
    except Exception as e:
        print(f"Error processing {imname}: {e}")

print("\nProcessing own images with pyramid alignment...")
for imname in own_image_files:
    try:
        output_name, offsets = process_pyramid_image(imname)
        own_output_files.append(output_name)
        own_offsets[imname] = offsets
    except Exception as e:
        print(f"Error processing {imname}: {e}")

# Save offsets to JSON files for HTML to read (optional, for backup)
with open('simple_image_offsets.json', 'w') as f:
    json.dump(simple_offsets, f, indent=2)

with open('simple_ncc_image_offsets.json', 'w') as f:
    json.dump(simple_ncc_offsets, f, indent=2)

with open('pyramid_image_offsets.json', 'w') as f:
    json.dump(pyramid_offsets, f, indent=2)

with open('own_image_offsets.json', 'w') as f:
    json.dump(own_offsets, f, indent=2)

# Embed offsets directly into HTML file
try:
    with open('index.html', 'r') as f:
        html_content = f.read()
    
    # Replace placeholder comments with actual JSON data
    html_content = html_content.replace(
        '/* OFFSET_DATA_SIMPLE */',
        json.dumps(simple_offsets, indent=8).replace('\n', '\n        ')
    )
    html_content = html_content.replace(
        '/* OFFSET_DATA_SIMPLE_NCC */',
        json.dumps(simple_ncc_offsets, indent=8).replace('\n', '\n        ')
    )
    html_content = html_content.replace(
        '/* OFFSET_DATA_PYRAMID */',
        json.dumps(pyramid_offsets, indent=8).replace('\n', '\n        ')
    )
    html_content = html_content.replace(
        '/* OFFSET_DATA_OWN */',
        json.dumps(own_offsets, indent=8).replace('\n', '\n        ')
    )
    
    with open('index.html', 'w') as f:
        f.write(html_content)
    
    print("\n✓ Offsets embedded directly into index.html")
except Exception as e:
    print(f"\nWarning: Could not embed offsets into HTML: {e}")
    print("  HTML file will use placeholder data. JSON files are still available.")

print(f"\nSimple alignment outputs (Euclidean): {simple_output_files}")
print(f"Simple alignment outputs (NCC): {simple_ncc_output_files}")
print(f"Pyramid alignment outputs: {pyramid_output_files}")
print(f"Own alignment outputs: {own_output_files}")
print(f"\nOffsets saved:")
print(f"  - Simple alignment offsets (Euclidean): simple_image_offsets.json")
print(f"  - Simple alignment offsets (NCC): simple_ncc_image_offsets.json")
print(f"  - Pyramid alignment offsets: pyramid_image_offsets.json")
print(f"  - Own image offsets: own_image_offsets.json")
print(f"  - Offsets also embedded in: index.html")