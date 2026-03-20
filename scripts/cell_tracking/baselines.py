"""
Classical optical flow baseline for comparison with VoxelMorph.

Horn & Schunck (1981) — first variational optical flow method.

All functions return displacement fields in VoxelMorph convention:
    disp[0] = dx (horizontal), disp[1] = dy (vertical), shape (2, H, W)
"""

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter


def horn_schunck(
    source: np.ndarray,
    target: np.ndarray,
    alpha: float = 1.0,
    num_iter: int = 100,
) -> np.ndarray:
    """
    Horn & Schunck (1981) optical flow.

    Minimizes: (Ix*u + Iy*v + It)^2 + alpha^2 * (|grad u|^2 + |grad v|^2)
    Solved iteratively via Gauss-Seidel with Laplacian averaging.

    Parameters
    ----------
    source, target : np.ndarray
        Grayscale images (H, W), float32, same size.
    alpha : float
        Smoothness weight. Higher = smoother flow.
    num_iter : int
        Number of iterations.

    Returns
    -------
    np.ndarray
        Displacement field (2, H, W): disp[0]=dx, disp[1]=dy.
    """
    # Image gradients (average of both frames for stability)
    avg = (source + target) / 2.0
    Ix = np.gradient(avg, axis=1)  # dI/dx
    Iy = np.gradient(avg, axis=0)  # dI/dy
    It = target - source           # dI/dt

    u = np.zeros_like(source)  # horizontal (dx)
    v = np.zeros_like(source)  # vertical (dy)

    alpha_sq = alpha ** 2

    for _ in range(num_iter):
        # Laplacian approximation via neighborhood average
        u_avg = uniform_filter(u, size=3) * (9 / 8) - u / 8
        v_avg = uniform_filter(v, size=3) * (9 / 8) - v / 8

        denom = alpha_sq + Ix ** 2 + Iy ** 2
        P = Ix * u_avg + Iy * v_avg + It

        u = u_avg - Ix * P / denom
        v = v_avg - Iy * P / denom

    return np.stack([u, v], axis=0).astype(np.float32)


def warp_image(image: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    Warp a 2D image using a displacement field (bilinear interpolation).

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W).
    displacement : np.ndarray
        Displacement field (2, H, W): disp[0]=dx, disp[1]=dy.

    Returns
    -------
    np.ndarray
        Warped image (H, W).
    """
    H, W = image.shape
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    coords = np.array([
        grid_y + displacement[1],  # row coords
        grid_x + displacement[0],  # col coords
    ])
    return map_coordinates(image, coords, order=1, mode='constant', cval=0.0).astype(np.float32)
