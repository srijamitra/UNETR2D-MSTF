"""
Training utilities for UNETR2D-MSTF.

This module contains the core loss functions, metrics,
learning-rate schedule, and training configuration used
for the segmentation framework.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100

LR_MAX = 1e-4
LR_MIN = 1e-6
WARMUP_EPOCHS = 10


# ============================================================
# Dice Coefficient
# ============================================================

def dice_coefficient(y_true, y_pred, num_classes=3):
    """
    Computes the mean Dice coefficient across segmentation classes.

    For REFUGE2, the background class is excluded from evaluation.
    """

    smooth = 1e-6

    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.one_hot(
        tf.argmax(y_pred, axis=-1),
        depth=num_classes
    )
    y_pred = tf.cast(y_pred, tf.float32)

    dice_scores = []

    for class_id in range(1, num_classes):
        true_class = K.flatten(y_true[..., class_id])
        pred_class = K.flatten(y_pred[..., class_id])

        intersection = K.sum(true_class * pred_class)

        dice = (
            2.0 * intersection + smooth
        ) / (
            K.sum(true_class) +
            K.sum(pred_class) +
            smooth
        )

        dice_scores.append(dice)

    return tf.reduce_mean(dice_scores)


# ============================================================
# IoU
# ============================================================

def iou_coefficient(y_true, y_pred, num_classes=3):
    """
    Computes the mean Intersection over Union (IoU)
    across foreground segmentation classes.
    """

    smooth = 1e-6

    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.one_hot(
        tf.argmax(y_pred, axis=-1),
        depth=num_classes
    )
    y_pred = tf.cast(y_pred, tf.float32)

    iou_scores = []

    for class_id in range(1, num_classes):
        true_class = K.flatten(y_true[..., class_id])
        pred_class = K.flatten(y_pred[..., class_id])

        intersection = K.sum(true_class * pred_class)

        union = (
            K.sum(true_class) +
            K.sum(pred_class) -
            intersection
        )

        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)

    return tf.reduce_mean(iou_scores)


# ============================================================
# Hybrid Loss
# ============================================================

def hybrid_loss(y_true, y_pred, num_classes=3):
    """
    Hybrid segmentation loss:

        Loss = 0.5 * Cross Entropy + 0.5 * Dice Loss

    Used with the deep-supervision prediction heads.
    """

    smooth = 1e-6

    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    y_true_one_hot = tf.one_hot(
        y_true,
        depth=num_classes
    )
    y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)

    # Categorical cross-entropy
    cross_entropy = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true_one_hot,
            y_pred
        )
    )

    # Dice loss
    dice_scores = []

    for class_id in range(num_classes):
        true_class = K.flatten(
            y_true_one_hot[..., class_id]
        )
        pred_class = K.flatten(
            y_pred[..., class_id]
        )

        intersection = K.sum(true_class * pred_class)

        dice = (
            2.0 * intersection + smooth
        ) / (
            K.sum(true_class) +
            K.sum(pred_class) +
            smooth
        )

        dice_scores.append(dice)

    dice_loss = 1.0 - tf.reduce_mean(dice_scores)

    return 0.5 * cross_entropy + 0.5 * dice_loss


# ============================================================
# Learning-Rate Schedule
# ============================================================

def cosine_with_warmup(epoch):
    """
    Linear warm-up followed by cosine decay.
    """

    if epoch < WARMUP_EPOCHS:
        return (
            LR_MIN +
            (LR_MAX - LR_MIN) *
            (epoch / WARMUP_EPOCHS)
        )

    progress = (
        (epoch - WARMUP_EPOCHS) /
        max(EPOCHS - WARMUP_EPOCHS, 1)
    )

    cosine_decay = 0.5 * (
        1.0 + np.cos(np.pi * progress)
    )

    return (
        LR_MIN +
        (LR_MAX - LR_MIN) * cosine_decay
    )


# ============================================================
# Optimizer
# ============================================================

def create_optimizer():
    """
    Creates the Adam optimizer used for training.
    """

    return tf.keras.optimizers.Adam(
        learning_rate=LR_MAX
    )


# ============================================================
# Training Configuration
# ============================================================

TRAINING_CONFIG = {
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate_max": LR_MAX,
    "learning_rate_min": LR_MIN,
    "warmup_epochs": WARMUP_EPOCHS,
    "optimizer": "Adam",
    "loss": "0.5 * Cross Entropy + 0.5 * Dice Loss",
}
