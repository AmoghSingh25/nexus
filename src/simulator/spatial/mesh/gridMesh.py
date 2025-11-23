from jax import random
import jax.numpy as jnp


class GridMesh:
    def __init__(self, height, width, depth, random_key=42):
        # Density - Cells(Entities) per unit volume
        # Each pos is positive - x>0, y>0, z>0

        assert width > 0 and height > 0, "Dimensions must be greater than 0"
        self.width = width
        self.height = height
        self.depth = depth

        self.key, self.sub_key = random.split(random.key(random_key))
        self.n_cells = int(self.height * self.width * self.depth)
        self.key, self.sub_key = random.split(self.key)

    def get_neighbours(self, idx):
        # 3D neighbours - 26 neighbours
        idx = jnp.array(idx)
        neighbor_idx = jnp.array(
            [
                [-1, -1, -1],
                [-1, -1, 0],
                [-1, -1, 1],
                [-1, 0, -1],
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, -1],
                [-1, 1, 0],
                [-1, 1, 1],
                [0, -1, -1],
                [0, -1, 0],
                [0, -1, 1],
                [0, 0, -1],
                [0, 0, 1],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],
                [1, -1, -1],
                [1, -1, 0],
                [1, -1, 1],
                [1, 0, -1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, -1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        neighbor_pos = []

        def check_positive(arr):
            for i in arr:
                if i < 0:
                    return False
            return True

        for i in neighbor_idx:
            pos_i = i + idx
            if check_positive(pos_i):
                neighbor_pos.append(pos_i)
        return neighbor_pos
