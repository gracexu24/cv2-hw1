# cv2-hw1-Grace-Xu

## Dependencies
- Python 3
- numpy
- scikit-image

## To run my code
1. Place all input images in the `code` directory
2. Required input images:
   - `cathedral.jpg`, `tobolsk.jpg`, `monastery.jpg`
   - `melons.tif`, `church.tif`, `emir.tif`, `harvesters.tif`, `icon.tif`, `italil.tif`, `lastochikino.tif`, `lugano.tif`, `self_portrait.tif`, `siren.tif`, `three_generations.tif`
   - Personal choice images: `ownimage1.tif`, `ownimage2.tif`, `ownimage3.tif` (rename your three images to these names)
3. In the `code` directory, run:
```bash
python main.py
```

## The script will generate: 
- Aligned images in the `code` directory (these are the same ones as in assets and used for index.html):
  - `*_simple_aligned.jpg` - Simple alignment results for cathedral, tobolsk, monastery
  - `*_pyramid_aligned.jpg` - Pyramid alignment results for all images
- JSON files with alignment offsets (was used for creating my index.html):
  - `simple_image_offsets.json`
  - `pyramid_image_offsets.json`
  - `own_image_offsets.json`
