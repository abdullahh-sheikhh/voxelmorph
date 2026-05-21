"""
Horn-Schunck optical flow with multi-scale pyramid strategy.

State-of-the-art implementation from Sorbonne Lip6 lab.
Uses iterative warping, SOR relaxation, and coarse-to-fine pyramid
for handling large displacements.

References:
    Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
        Artificial intelligence, 17(1-3), 185-203.
    Meinhardt-Llopis, E., & Sanchez, J. (2013). Horn-Schunck optical flow
        with a multi-scale strategy. Image Processing On Line.
"""

from typing import Union
import numpy as np
from scipy.signal import convolve
from scipy import ndimage as ndi
from skimage.transform import pyramid_reduce
from scipy.ndimage import map_coordinates


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warps a multi dimensionnal image using an optical flow

    Args:
        image (np.ndarray): Image to warp. Can optionnally have a channel dimension.
            Shape: (H, ...[, C])
        flow (np.ndarray): Optical flow. The first dimension is 2 for 2D images, 3 for 3D images, etc...
            Shape: (ndim, H, ...), dtype: float (32 or 64)

    Returns:
        np.ndarray: Warped image
            Shape: (H, ...[, C])
    """
    assert (
        flow.ndim - 1 == flow.shape[0]
    ), "First dimension should match the number of trailing dimensions"

    if image.ndim == flow.ndim:
        *dims, channels = image.shape
        points = np.indices(dims, flow.dtype) + flow

        return np.concatenate(
            [
                map_coordinates(image[..., channel], points, mode='nearest', order=1)[
                    ..., None
                ]
                for channel in range(channels)
            ],
            axis=-1,
        )

    # No channels
    assert image.ndim == flow.ndim - 1

    return map_coordinates(
        image, np.indices(image.shape, flow.dtype) + flow, mode='nearest', order=1
    )


def resize_flow(flow, shape):
    """
    Rescales the values of the vector field (u, v) to the desired shape.
    The values of the output vector field are scaled to the new resolution.

    Args:
        flow (np.ndarray): The displacement field
            Shape: (ndim, H, W, ...)
        shape (iterable): Couple of integers representing the output shape

    Returns:
        np.ndarray: The resized and rescaled motion field
    """

    scale = [n / o for n, o in zip(shape, flow.shape[1:])]
    scale_factor = np.array(scale, dtype=flow.dtype)

    for _ in shape:
        scale_factor = scale_factor[..., np.newaxis]

    rflow = scale_factor * ndi.zoom(
        flow, [1] + scale, order=1, mode="nearest", prefilter=False
    )

    return rflow


def get_pyramid(image, downscale=2.0, nlevel=10, min_size=16):
    """
    Construct image pyramid.

    Args:
        image (np.ndarray): The image to be preprocessed (Gray scale or RGB)
            Shape: (H, W, ...)
        downscale (float): The pyramid downscale factor
            Default: 2
        nlevel (int): The maximum number of pyramid levels
            Default: 10
        min_size (int): The minimum size for any dimension of the pyramid levels
            Default: 16

    Returns :
        list[ndarray]: The coarse to fine images pyramid
    """

    pyramid = [image]
    size = min(image.shape)
    count = 1

    while (count < nlevel) and (size > downscale * min_size):
        J = pyramid_reduce(pyramid[-1], downscale)
        pyramid.append(J)
        size = min(J.shape)
        count += 1

    return pyramid[::-1]


def _create_average_kernel(dimension: int):
    """
    Creates an average kernel for a specified dimension. The values of the discrete
    kernel for 2D can be found in the seminal paper by Horn-Schunck.

    Args:
        dimension (int): The dimension of the kernel. Acceptable values are:
                         1 (for a 1D kernel), 2 (for a 2D kernel), or 3 (for a 3D kernel).

    Returns:
        np.ndarray: A NumPy array representing the averaging kernel for the specified dimension.

    References:
        Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
            Artificial intelligence, 17(1-3), 185-203
    """
    # Create a 1D average kernel if dimension is 1
    if dimension == 1:
        # Returns a 1D array where non-zero elements are set to 1/6
        return np.array([1 / 2, 0, 1 / 2], dtype=np.float32)

    # Create a 2D average kernel if dimension is 2
    elif dimension == 2:
        return np.array(
            [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]],
            dtype=np.float32,
        )

    # Create a 3D average kernel if dimension is 3
    elif dimension == 3:
        return np.array(
            [
                [
                    [1 / 24, 1 / 12, 1 / 24],
                    [1 / 12, 1 / 6, 1 / 12],
                    [1 / 24, 1 / 12, 1 / 24],
                ],
                [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]],
                [
                    [1 / 24, 1 / 12, 1 / 24],
                    [1 / 12, 1 / 6, 1 / 12],
                    [1 / 24, 1 / 12, 1 / 24],
                ],
            ],
            dtype=np.float32,
        )

    else:
        raise ValueError("Dimension must be 1, 2, or 3.")


def _hs_optical_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    u0: np.ndarray,
    alpha: float,
    num_iter=100,
    num_warp=2,
    eps=1e-5,
    w=1.0,
    dtype=np.float32,
):
    """
    Computes the optical flow between two images (reference and moving) using
    the Horn-Schunck (HS) algorithm with iterative warping.

    This algorithm applies to small displacements only and shall be paired with
    a multiscale strategy for larger displacements.

    The sparse system ensuing from the resolution of the discrete Euler-Lagrange equations is solved
    by Successive Over-Relaxation (SOR), which generalizes the Jacobi iteration initially presented in the
    seminal paper of Horn and Schunck.

    Args:
        reference (np.ndarray): The reference (static) image to which the moving image is aligned.
            Shape: (M, N,...)
        moving (np.ndarray): The moving image that is being warped towards the reference.
            Shape: (M, N,...)
        u0 (np.ndarray): The initial displacement field guess.
            Shape: (d, M, N,...)
        alpha (float): Regularization parameter controlling smoothness of the flow field.
        num_iter (int): Number of iterations for the HS algorithm.
            Default: 100
        num_warp (int, optional): Number of warping steps for iterative refinement.
            Default: 2
        eps (float, optional): Convergence tolerance. Algorithm stops if mean squared change in
                               displacement field is below this threshold.
            Default: 1e-5.
        w (float, optional): Relaxation parameter for SOR.  Values shall be in (0,2).
                            A value of 1 corresponds to Jacobi iteration.
            Default: 1.
        dtype (np.dtype, optional): Data type for computation (default is np.float32).

    Returns:
        np.ndarray: The computed displacement field (optical flow) that warps the moving image
                    towards the reference image, with shape matching u0.

    References:
        Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
            Artificial intelligence, 17(1-3), 185-203
    """
    # Input images must be of the same shape
    assert moving.shape == reference.shape

    # Check if relaxation parameter is in the admissible range
    if not 0 < w < 2:
        raise ValueError(f"Over relaxation parameter should be in (0,2). Found: {w}")

    # Dimension of input images (1D, 2D or 3D), assumed from reference image
    dim = reference.ndim

    # Cast input arrays to specified dtype for consistency in computation
    u0 = u0.astype(dtype)
    reference = reference.astype(dtype)
    moving = moving.astype(dtype)

    # Initialize displacement field u as zero matrix with same shape as u0
    u = u0.copy()

    # Create a Laplacian kernel for smoothing the flow field, based on image dimensions
    laplace_kernel = _create_average_kernel(reference.ndim)
    laplace_kernel /= np.sum(laplace_kernel)

    # Iterative warping loop, to refine the displacement field (optical flow)
    for _ in range(num_warp):
        # Warp the moving image according to the current displacement estimate u0
        im2_warped = warp(moving, u0)

        # Compute spatial gradients of the warped image
        nabla_I = np.array(np.gradient(im2_warped))

        # Compute temporal intensity difference between reference and warped image
        im_t = im2_warped - reference

        # Inner loop for the HS algorithm to iteratively refine the flow field
        for _ in range(num_iter):
            # Compute smoothed version of the current displacement field u using convolution
            u_average = convolve(
                np.pad(
                    u, pad_width=((0, 0),) + tuple([tuple((1, 1))] * dim), mode="edge"
                ),
                laplace_kernel[None, ...],
                mode="valid",
            )

            # Derivative-based term used to update the displacement field, balancing data and smoothness
            der = ((nabla_I * (u_average - u0)).sum(axis=0) + im_t) / (
                (nabla_I * nabla_I).sum(axis=0) + alpha
            )

            # Update displacement field by subtracting gradient-scaled derivative
            u_new = u_average - nabla_I * der

            # Convergence check based on mean squared change in u
            if np.mean((u_new - u) ** 2) < eps**2:
                break

            # Update u with a weighted relaxation of u_new for stability and convergence control
            u = w * u_new + (1 - w) * u

        # Update initial displacement field for next warping step
        u0 = np.array(u, dtype=dtype)

    return u0


def hs_optical_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    alpha: float,
    num_iter=100,
    num_warp=2,
    num_pyramid=10,
    pyramid_downscale: Union[float, np.ndarray] = 2.0,
    pyramid_min_size: Union[int, np.ndarray] = 16,
    eps=1e-5,
    w=1.0,
):
    """
    Computes optical flow between two images (reference and moving) using the
    Horn-Schunck (HS) algorithm with a multi-scale image pyramid for efficient
    large-displacement handling.

    Args:
        reference (np.ndarray): The reference (static) image to which the moving image is aligned.
            Shape: (M, N,...)
        moving (np.ndarray): The moving image to be aligned with the reference image.
            Shape: (M, N,...)
        alpha (float): Regularization parameter that controls the smoothness of the computed flow.
        num_iter (int, optional): Number of iterations at each pyramid level for the HS algorithm.
            Default: 100
        num_warp (int, optional): Number of warping steps for each pyramid level.
            Default: 2
        num_pyramid (int, optional): Number of pyramid levels for multi-scale processing.
            Default: 10
        pyramid_downscale (Union[float, np.ndarray], optional): Scaling factor or array for downsampling
                             at each pyramid level.
            Default: 2
        pyramid_min_size (Union[int, np.ndarray], optional): Minimum size for images in the pyramid to stop downscaling.
            Default: 16
        eps (float, optional): Convergence threshold based on mean squared change in flow field.
            Default: 1e-5
        w (float, optional): Relaxation parameter for SOR.  Values shall be in (0,2).
                            A value of 1 corresponds to Jacobi iteration.
            Default: 1

    Returns:
        np.ndarray: The computed displacement field (optical flow) that aligns the moving image to the reference image.

    References:
        Meinhardt-Llopis, E., & Sanchez, J. (2013). Horn-schunck optical flow with a multi-scale strategy.
        Image Processing on line.
    """
    # Input images must be of the same shape
    assert moving.shape == reference.shape

    # Check if relaxation parameter is in the admissible range
    if not 0 < w < 2:
        raise ValueError(f"Over relaxation parameter should be in (0,2). Found: {w}")

    # Dimension of the input images (e.g., 2D or 3D)
    d = reference.ndim

    # Build the image pyramids for both reference and moving images
    pyramid_ = list(
        zip(
            get_pyramid(reference, pyramid_downscale, num_pyramid, pyramid_min_size),
            get_pyramid(moving, pyramid_downscale, num_pyramid, pyramid_min_size),
        )
    )

    # Initialize displacement field (optical flow) at the coarsest pyramid level
    u0 = np.zeros(
        (d,) + pyramid_[0][0].shape
    )  # Shape matching the coarsest reference level

    # Compute optical flow at the coarsest pyramid level
    u = _hs_optical_flow(
        pyramid_[0][0],
        pyramid_[0][1],
        u0,
        alpha,
        num_iter=num_iter,
        num_warp=num_warp,
        eps=eps,
        w=w,
    )

    # Progressively refine the optical flow up the pyramid levels
    for pyr_ref, pyr_mov in pyramid_[1:]:
        # Resize the flow field to the current pyramid level's dimensions
        u = resize_flow(u, pyr_ref.shape)

        # Update the flow field using the HS algorithm at the current pyramid level
        u = _hs_optical_flow(
            pyr_ref,
            pyr_mov,
            u,
            alpha,
            num_iter=num_iter,
            num_warp=num_warp,
            eps=eps,
            w=w,
        )

    return u
