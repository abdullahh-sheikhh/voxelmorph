import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter


def horn_schunck(source, target, alpha=1.0, num_iter=100):
    """Estimate displacement field between source and target using Horn & Schunck."""
    avg = (source + target) / 2.0
    Ix = np.gradient(avg, axis=1)
    Iy = np.gradient(avg, axis=0)
    It = source - target

    u = np.zeros_like(source)
    v = np.zeros_like(source)
    alpha_sq = alpha ** 2

    for _ in range(num_iter):
        u_avg = uniform_filter(u, size=3) * (9 / 8) - u / 8
        v_avg = uniform_filter(v, size=3) * (9 / 8) - v / 8

        denom = alpha_sq + Ix ** 2 + Iy ** 2
        P = Ix * u_avg + Iy * v_avg + It

        u = u_avg - Ix * P / denom
        v = v_avg - Iy * P / denom

    return np.stack([u, v], axis=0).astype(np.float32)


def warp_image(image, displacement):
    """Warp image using displacement field with bilinear interpolation."""
    H, W = image.shape
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = np.array([
        grid_y + displacement[1],
        grid_x + displacement[0],
    ])
    return map_coordinates(image, coords, order=1, mode='constant', cval=0.0).astype(np.float32)
