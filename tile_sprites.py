
import isometric_projection.transforms as iso_t
import utils.sprites as sp
import matplotlib.pyplot as plt
import numpy as np
import world_creation.bitmap_world as bmw


#build isometric terrain square with a given size
def build_grid(side_len, scale):

    # load sprites
    dirt = sp.load_sprite("/home/twovans/Pictures/IsoSprites/16px/Sheets/GrassFrames.png", (16, 20))
    rock = sp.load_sprite("/home/twovans/Pictures/IsoSprites/16px/Sheets/RockFrames.png", (16, 20))
    water_flat = sp.load_sprite("/home/twovans/Pictures/IsoSprites/16px/Sheets/waterFlat.png", (16, 20))
    sprite = dirt + rock + water_flat
    sprite_dims = sprite[0].shape

    #compute the terrain
    height_grid = bmw.world_gen([side_len] * 2, scale = scale)

    #allocate the image array
    grid_img = 255 * np.ones((10*side_len, 16*side_len, 4), dtype=np.uint8)
    
    #add blocks to the world
    for row in range(side_len):
        for col in range(side_len):

            #pull elevation
            z = (height_grid[row, col] - 0.4)

            #determine sprite type
            if (z > 0.3): sprite_version = (row * col)%4 + 4
            elif (z > -0.1): sprite_version = (row * col)%4
            else:
                z = -0.1
                sprite_version = (row*col)%4 + 8

            #scale z to world coordinates
            z *= side_len

            # project the x and y and z coordinates to isometric projections
            top_left_x, top_left_y = iso_t.xyz_to_iso_xy(16*col, 16*row, z)


            top_left_y += side_len
            top_left_x += grid_img.shape[1]//2-sprite_dims[1]//2
            
            #blend the sprite into the image (need to figure out how to do this faster)
            sp.opacity_blending_add(sprite[sprite_version], grid_img, (top_left_x, top_left_y))

        # print progress
        print(f"\r {(row*100)//side_len}%", end = "")
            
    return grid_img

if __name__ == "__main__":

    side = 500
    scale = 1.8
    image = build_grid(side, scale)
    plt.imsave(f"isoWorld{side}x{side}Scale{scale}.png", image)
    plt.imshow(image)
    plt.show()

            


