"""
UNETR2D-MSTF
Preprocessing and Data Pipeline

The datasets itself is not included in this repository.
"""

import numpy as np
import tensorflow as tf
from scipy import ndimage
import cv2


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
BATCH_SIZE = 16


# ============================================================
# Augmentation 1: Random Affine / Rotation
# ============================================================

def random_affine(image, mask):
    """Apply a random rotation to an image and segmentation mask."""

    angle = tf.random.uniform([], -0.26, 0.26)

    def _affine_np(img_np, msk_np, angle_val):

        img_np = np.array(img_np)
        msk_np = np.array(msk_np)

        h = img_np.shape[0]
        cy, cx = (h - 1) / 2.0, (h - 1) / 2.0

        out_y, out_x = np.mgrid[0:h, 0:h].astype(np.float32)

        cos_a = np.cos(angle_val)
        sin_a = np.sin(angle_val)

        src_x = (
            cos_a * (out_x - cx)
            + sin_a * (out_y - cy)
            + cx
        )

        src_y = (
            -sin_a * (out_x - cx)
            + cos_a * (out_y - cy)
            + cy
        )

        coords = np.array([
            src_y.ravel(),
            src_x.ravel()
        ])

        # Bilinear interpolation for RGB image
        img_aug = np.zeros_like(img_np)

        for c in range(3):
            img_aug[..., c] = ndimage.map_coordinates(
                img_np[..., c],
                coords,
                order=1,
                mode="reflect"
            ).reshape(h, h)

        # Nearest-neighbour interpolation for mask
        msk_sq = np.squeeze(msk_np)

        msk_aug = ndimage.map_coordinates(
            msk_sq,
            coords,
            order=0,
            mode="constant",
            cval=0
        ).reshape(h, h, 1)

        return (
            img_aug.astype(np.float32),
            msk_aug.astype(np.float32)
        )

    img, msk = tf.py_function(
        _affine_np,
        [image, mask, angle],
        [tf.float32, tf.float32]
    )

    img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    msk.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])

    return img, msk


# ============================================================
# Augmentation 2: Random Flips
# ============================================================

def random_flips(image, mask):
    """Apply random horizontal and vertical flips."""

    image, mask = tf.cond(
        tf.random.uniform(()) > 0.5,
        lambda: (
            tf.image.flip_left_right(image),
            tf.image.flip_left_right(mask)
        ),
        lambda: (image, mask)
    )

    image, mask = tf.cond(
        tf.random.uniform(()) > 0.5,
        lambda: (
            tf.image.flip_up_down(image),
            tf.image.flip_up_down(mask)
        ),
        lambda: (image, mask)
    )

    return image, mask


# ============================================================
# Augmentation 3: Elastic Deformation
# ============================================================

def elastic_deformation(
    image,
    mask,
    alpha=20.0,
    sigma=5.0
):
    """Apply random elastic deformation to image and mask."""

    def _elastic_np(img_np, msk_np):

        img_np = np.array(img_np)
        msk_np = np.array(msk_np)

        h = img_np.shape[0]

        dx = ndimage.gaussian_filter(
            np.random.randn(h, h) * alpha,
            sigma,
            mode="reflect"
        )

        dy = ndimage.gaussian_filter(
            np.random.randn(h, h) * alpha,
            sigma,
            mode="reflect"
        )

        y, x = np.meshgrid(
            np.arange(h),
            np.arange(h),
            indexing="ij"
        )

        coords = np.array([
            (y + dy).ravel(),
            (x + dx).ravel()
        ])

        # Image interpolation
        img_aug = np.zeros_like(img_np)

        for c in range(3):
            img_aug[..., c] = ndimage.map_coordinates(
                img_np[..., c],
                coords,
                order=1,
                mode="reflect"
            ).reshape(h, h)

        # Mask interpolation
        msk_sq = np.squeeze(msk_np)

        msk_aug = ndimage.map_coordinates(
            msk_sq,
            coords,
            order=0,
            mode="constant",
            cval=0
        ).reshape(h, h, 1)

        return (
            img_aug.astype(np.float32),
            msk_aug.astype(np.float32)
        )

    def _apply():

        img_out, msk_out = tf.py_function(
            _elastic_np,
            [image, mask],
            [tf.float32, tf.float32]
        )

        img_out.set_shape([
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ])

        msk_out.set_shape([
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        ])

        return img_out, msk_out

    return tf.cond(
        tf.random.uniform(()) > 0.5,
        _apply,
        lambda: (image, mask)
    )


# ============================================================
# Augmentation 4: Safe Random Zoom
# ============================================================

def _safe_zoom_fn(img_np, msk_np, scale_val):

    img = np.array(img_np, dtype=np.float32)
    msk = np.array(msk_np, dtype=np.float32)

    scale = float(scale_val)

    h = img.shape[0]

    new_size = int(h * scale)
    new_size = max(16, min(new_size, h - 1))

    start = (h - new_size) // 2

    img_crop = img[
        start:start + new_size,
        start:start + new_size,
        :
    ]

    msk_crop = msk[
        start:start + new_size,
        start:start + new_size,
        :
    ]

    img_zoom = cv2.resize(
        img_crop,
        (h, h),
        interpolation=cv2.INTER_LINEAR
    )

    msk_zoom_2d = cv2.resize(
        msk_crop[..., 0],
        (h, h),
        interpolation=cv2.INTER_NEAREST
    )

    msk_zoom = (
        msk_zoom_2d[..., np.newaxis]
        .astype(np.float32)
    )

    return (
        img_zoom.astype(np.float32),
        msk_zoom
    )


def safe_random_zoom(image, mask):
    """Apply a random zoom while preserving output dimensions."""

    scale = tf.random.uniform(
        [],
        0.85,
        0.98
    )

    def _apply():

        img_out, msk_out = tf.py_function(
            _safe_zoom_fn,
            [image, mask, scale],
            [tf.float32, tf.float32]
        )

        img_out.set_shape([
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ])

        msk_out.set_shape([
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        ])

        return img_out, msk_out

    return tf.cond(
        tf.random.uniform(()) > 0.6,
        _apply,
        lambda: (image, mask)
    )


# ============================================================
# Augmentation 5–8: Intensity Augmentations
# ============================================================

def apply_intensity_augs(image):
    """Apply random brightness, contrast, noise and gamma changes."""

    # Random brightness
    image = tf.cond(
        tf.random.uniform(()) > 0.5,
        lambda: tf.image.random_brightness(
            image,
            0.15
        ),
        lambda: image
    )

    # Random contrast
    image = tf.cond(
        tf.random.uniform(()) > 0.5,
        lambda: tf.image.random_contrast(
            image,
            0.8,
            1.2
        ),
        lambda: image
    )

    # Gaussian noise
    image = tf.cond(
        tf.random.uniform(()) > 0.7,
        lambda: image + tf.random.normal(
            shape=tf.shape(image),
            mean=0.0,
            stddev=0.02
        ),
        lambda: image
    )

    # Gamma adjustment
    gamma = tf.random.uniform(
        [],
        0.7,
        1.5
    )

    image = tf.cond(
        tf.random.uniform(()) > 0.6,
        lambda: tf.pow(
            tf.clip_by_value(
                image,
                1e-7,
                1.0
            ),
            gamma
        ),
        lambda: image
    )

    return tf.clip_by_value(
        image,
        0.0,
        1.0
    )


# ============================================================
# Complete Augmentation Pipeline
# ============================================================

def safe_apply_augmentations(image, mask):
    """
    Apply the complete training augmentation pipeline.

    The same spatial transformations are applied to the
    image and segmentation mask.
    """

    image, mask = random_affine(
        image,
        mask
    )

    image, mask = random_flips(
        image,
        mask
    )

    image, mask = elastic_deformation(
        image,
        mask,
        alpha=20.0,
        sigma=5.0
    )

    image, mask = safe_random_zoom(
        image,
        mask
    )

    image = apply_intensity_augs(image)

    # Preserve discrete segmentation labels
    mask = tf.math.round(mask)

    return image, mask


# ============================================================
# Image and Mask Loading
# ============================================================

def load_data(image_path, mask_path):
    """
    Load and preprocess an ISIC 2018 image-mask pair.

    Images:
        RGB JPEG, normalized to [0, 1]

    Masks:
        Grayscale PNG
        0   -> background
        255 -> lesion
    """

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    img = tf.io.read_file(image_path)

    img = tf.io.decode_image(
        img,
        channels=3,
        expand_animations=False
    )

    img = tf.image.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    img = tf.cast(
        img,
        tf.float32
    ) / 255.0

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    msk = tf.io.read_file(mask_path)

    msk = tf.io.decode_image(
        msk,
        channels=1,
        expand_animations=False
    )

    msk = tf.image.resize(
        msk,
        (IMAGE_SIZE, IMAGE_SIZE),
        method="nearest"
    )

    # 255 -> lesion (1)
    # 0   -> background (0)
    m = tf.cast(
        msk,
        tf.int32
    )

    new_mask = tf.where(
        m >= 128,
        1,
        0
    )

    return (
        img,
        tf.cast(
            new_mask,
            tf.float32
        )
    )


# ============================================================
# Patch Extraction
# ============================================================

def image_to_patches(image):
    """
    Convert a 256x256 RGB image into 16x16 non-overlapping patches.

    Output:
        256 patches
        16x16x3 = 768 values per patch
    """

    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, 0),
        sizes=[
            1,
            PATCH_SIZE,
            PATCH_SIZE,
            1
        ],
        strides=[
            1,
            PATCH_SIZE,
            PATCH_SIZE,
            1
        ],
        rates=[
            1,
            1,
            1,
            1
        ],
        padding="VALID"
    )

    patches = tf.reshape(
        patches,
        (
            NUM_PATCHES,
            PATCH_SIZE * PATCH_SIZE * 3
        )
    )

    return patches


# ============================================================
# Dataset Preprocessing
# ============================================================

def preprocess_train(image_path, mask_path):
    """
    Training preprocessing:
    loading -> augmentation -> patch extraction.
    """

    img, msk = load_data(
        image_path,
        mask_path
    )

    img, msk = safe_apply_augmentations(
        img,
        msk
    )

    patches = image_to_patches(img)

    # Same ground truth is supplied to each
    # deep-supervision prediction head.
    return patches, (
        msk,
        msk,
        msk,
        msk
    )


def preprocess_val(image_path, mask_path):
    """
    Validation/test preprocessing without augmentation.
    """

    img, msk = load_data(
        image_path,
        mask_path
    )

    patches = image_to_patches(img)

    return patches, (
        msk,
        msk,
        msk,
        msk
    )
