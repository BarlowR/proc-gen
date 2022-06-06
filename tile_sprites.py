
import isometric_projection.transforms as iso_t
import utils.sprites as sp
import matplotlib.pyplot as plt
import numpy as np
import world_creation.bitmap_world as bmw


#build a 4*4 isometric grid by sequentially selecting verions of the passed sprite 
def build_grid(sprite, side_len):

    sprite_dims = sprite[0].shape

    height_grid = bmw.world_gen([side_len] * 2, scale = 4)

    grid_img = 255 * np.ones((10*side_len, 16*side_len, 4), dtype=np.uint8)
    for row in range(side_len):
        for col in range(side_len):

            y = side_len*1.5 * (height_grid[row, col] - 0.3)
            if (y > 0): sprite_version = (row * col)%4
            else: sprite_version = (row * col)%4 + 4
            top_left_x, top_left_y = iso_t.xyz_to_iso_xy(16*col, 16*row, y)
            top_left_x += grid_img.shape[1]//2-sprite_dims[1]//2
            
            sp.opacity_blending_add(sprite[sprite_version], grid_img, (top_left_x, top_left_y))
            
    return grid_img

if __name__ == "__main__":

    dirt = sp.load_sprite("/home/twovans/Pictures/IsoSprites/16px/Sheets/GrassFrames.png", (16, 20))
    rock = sp.load_sprite("/home/twovans/Pictures/IsoSprites/16px/Sheets/RockFrames.png", (16, 20))

    plt.imshow(build_grid(dirt+rock, 16))
    plt.show()

            


