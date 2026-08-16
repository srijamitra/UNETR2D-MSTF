"""
UNETR2D-MSTF
Inference and Test-Time Augmentation

Public implementation of the inference strategy described
in the UNETR2D-MSTF research work.

The inference pipeline uses:
- Test-Time Augmentation (TTA)
- Geometric transformations
- Brightness variations
- STAPLE probabilistic label fusion

The datasets and trained model weights are not included.
"""

import numpy as np
import tensorflow as tf
import SimpleITK as sitk


IMAGE_SIZE = 256


# ============================================================
# Basic Image Transformations
# ============================================================

def rotate_image(image, k):
    """
    Rotate an image by k * 90 degrees.
    """

    return np.rot90(
        image,
        k=k,
        axes=(0, 1)
    ).copy()


def flip_horizontal(image):
    """Flip image along the horizontal axis."""
    return np.flip(
        image,
        axis=1
    ).copy()


def flip_vertical(image):
    """Flip image along the vertical axis."""
    return np.flip(
        image,
        axis=0
    ).copy()


def adjust_brightness(image, factor):
    """
    Apply a brightness adjustment.

    The image is assumed to be normalized to [0, 1].
    """

    image = image * factor

    return np.clip(
        image,
        0.0,
        1.0
    )


# ============================================================
# TTA Transformations
# ============================================================

def generate_tta_images(image):
    """
    Generate test-time augmented versions of an image.

    The documented research protocol uses geometric
    transformations and brightness variations.

    Returns:
        List of transformed images and the corresponding
        inverse-transform functions.
    """

    tta_images = []
    inverse_transforms = []

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    tta_images.append(image.copy())
    inverse_transforms.append(
        lambda x: x
    )

    # --------------------------------------------------------
    # Rotations: 90, 180 and 270 degrees
    # --------------------------------------------------------

    for k in [1, 2, 3]:

        tta_images.append(
            rotate_image(image, k)
        )

        inverse_transforms.append(
            lambda x, k=k:
            rotate_image(x, 4 - k)
        )

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    tta_images.append(
        flip_horizontal(image)
    )

    inverse_transforms.append(
        lambda x:
        flip_horizontal(x)
    )

    # --------------------------------------------------------
    # Vertical flip
    # --------------------------------------------------------

    tta_images.append(
        flip_vertical(image)
    )

    inverse_transforms.append(
        lambda x:
        flip_vertical(x)
    )

    # --------------------------------------------------------
    # Brightness variations
    # --------------------------------------------------------

    for factor in [0.8, 1.2]:

        tta_images.append(
            adjust_brightness(
                image,
                factor
            )
        )

        inverse_transforms.append(
            lambda x:
            x
        )

    return tta_images, inverse_transforms


# ============================================================
# Model Prediction
# ============================================================

def predict_single_image(model, image):
    """
    Generate a segmentation prediction for one image.

    The model is expected to accept the tokenized input
    produced by the preprocessing pipeline.
    """

    prediction = model.predict(
        image,
        verbose=0
    )

    # Deep-supervision models return multiple outputs.
    # The final prediction is used for inference.
    if isinstance(prediction, list):
        prediction = prediction[-1]

    return prediction


# ============================================================
# TTA Prediction
# ============================================================

def predict_with_tta(
    model,
    image,
    preprocess_fn
):
    """
    Generate predictions using Test-Time Augmentation.

    Each transformed image is passed through the model.
    Predictions are transformed back to the original
    orientation before fusion.

    Returns:
        List of class-wise probability maps.
    """

    tta_images, inverse_transforms = (
        generate_tta_images(image)
    )

    predictions = []

    for augmented_image, inverse_transform in zip(
        tta_images,
        inverse_transforms
    ):

        # Apply the same tokenization/preprocessing
        # used during model inference.
        model_input = preprocess_fn(
            augmented_image
        )

        prediction = predict_single_image(
            model,
            model_input
        )

        # Convert class probabilities into spatial format
        if prediction.ndim == 4:

            prediction = prediction[0]

        prediction = inverse_transform(
            prediction
        )

        predictions.append(
            prediction
        )

    return predictions


# ============================================================
# STAPLE Fusion
# ============================================================

def staple_fusion(
    predictions,
    num_classes
):
    """
    Fuse TTA predictions using STAPLE.

    STAPLE is applied independently to each class.
    The resulting probabilistic maps are thresholded
    to obtain the final segmentation.
    """

    predictions = np.asarray(
        predictions
    )

    height = predictions.shape[1]
    width = predictions.shape[2]

    fused_probabilities = np.zeros(
        (
            height,
            width,
            num_classes
        ),
        dtype=np.float32
    )

    for class_id in range(num_classes):

        class_predictions = (
            predictions[..., class_id]
        )

        # Convert each probability map to a binary
        # mask for STAPLE fusion.
        binary_masks = (
            class_predictions >= 0.5
        ).astype(np.uint8)

        # Convert to SimpleITK images.
        sitk_masks = [
            sitk.GetImageFromArray(mask)
            for mask in binary_masks
        ]

        # STAPLE probabilistic fusion.
        staple_filter = sitk.STAPLEImageFilter()

        probability_image = (
            staple_filter.Execute(
                sitk_masks
            )
        )

        probability_map = (
            sitk.GetArrayFromImage(
                probability_image
            )
        )

        fused_probabilities[
            ...,
            class_id
        ] = probability_map

    return fused_probabilities


# ============================================================
# Final Segmentation
# ============================================================

def final_segmentation(
    fused_probabilities
):
    """
    Convert fused class probabilities into a
    final class-label segmentation mask.
    """

    return np.argmax(
        fused_probabilities,
        axis=-1
    ).astype(np.uint8)


# ============================================================
# Complete Inference Pipeline
# ============================================================

def run_inference(
    model,
    image,
    preprocess_fn,
    num_classes
):
    """
    Complete UNETR2D-MSTF inference pipeline:

        Input image
              ↓
        TTA transformations
              ↓
        Model predictions
              ↓
        Inverse transformations
              ↓
        STAPLE fusion
              ↓
        Final segmentation
    """

    predictions = predict_with_tta(
        model,
        image,
        preprocess_fn
    )

    fused_probabilities = staple_fusion(
        predictions,
        num_classes
    )

    segmentation = final_segmentation(
        fused_probabilities
    )

    return segmentation
