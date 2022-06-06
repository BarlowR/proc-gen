import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# loads equal sized sprites from an image file into an list of numpy image data arrays
#file_path: relative path to sprite sheet
def load_sprite(file_path, sprite_dims):
    sprite_sheet = np.array(Image.open(file_path))

    sprite_sheet_dims = sprite_sheet.shape

    sprite_cols = sprite_sheet_dims[0]//sprite_dims[1]
    sprite_rows = sprite_sheet_dims[1]//sprite_dims[0]

    sprites = []

    for row in range(sprite_rows):
        for col in range(sprite_cols):
            start_x = row * sprite_dims[0]
            start_y = col * sprite_dims[1]

            sprites.append(sprite_sheet[start_y:start_y+sprite_dims[1], start_x:start_x+sprite_dims[0]])
    return sprites

def opacity_blending_add(sprite, canvas, position):

    sprite_dims = sprite.shape

    for prow in range(sprite_dims[0]):
        for pcol in range(sprite_dims[1]):
            
            sprite_pixel = sprite[prow, pcol]
            canvas_pixel = canvas[prow + position[1], pcol + position[0]]
            R1,G1,B1,_ = canvas_pixel
            R2,G2,B2,OP = sprite_pixel 
            OP *= 1/255.0
            R3 = (int)((1-OP)*R1 + OP*R2)
            G3 = (int)((1-OP)*G1 + OP*G2)
            B3 = (int)((1-OP)*B1 + OP*B2)

            canvas[prow + position[1], pcol + position[0]] = [R3, G3, B3, 255]

def test_load_sprite():
    dirt = load_sprite("/home/twovans/Pictures/IsoSprites/16px/Sheets/GrassFrames.png", (16, 20))
    plt.imshow(dirt[0])
    plt.show()

