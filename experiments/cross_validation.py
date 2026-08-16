"""
UNETR2D-MSTF
5-Fold Cross-Validation — REFUGE2

Each fold:
    1. Creates a fresh UNETR2D-MSTF model
    2. Uses 4 folds for training
    3. Uses 1 fold for validation
    4. Trains from scratch
    5. Saves the best checkpoint according to validation mean Dice

The implementation follows the REFUGE2 cross-validation notebook.
"""

import os
import gc
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
)
import tensorflow.keras.backend as K


# ============================================================
# CONFIGURATION
# ============================================================

CF = {
    "image_size": 256,
    "num_channels": 3,
    "patch_size": 16,
    "num_patches": (256 // 16) ** 2,
    "hidden_dim": 512,
    "mlp_dim": 2048,
    "num_layers": 12,
    "num_query_heads": 8,
    "head_dim": 64,
    "dropout_rate": 0.1,
}

IMAGE_SIZE = CF["image_size"]
PATCH_SIZE = CF["patch_size"]

BATCH_SIZE = 16
EPOCHS = 35

LR_MAX = 1e-4
LR_MIN = 1e-6
WARMUP_EPOCHS = 10

N_SPLITS = 5
RANDOM_STATE = 42

CHECKPOINT_DIR = "checkpoints/cv"


# ============================================================
# MSTF
# ============================================================

def conv_block(x, filters):
    x = L.Conv2D(
        filters,
        3,
        padding="same"
    )(x)

    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    return x


def MSTF(x_tokens, cf, use_local=True,
         use_global=True, use_dilated=True):

    H = cf["image_size"] // cf["patch_size"]
    D = cf["hidden_dim"]

    x = L.Reshape(
        (H, H, D)
    )(x_tokens)

    branches = []

    # Local branch
    if use_local:
        local = conv_block(
            x,
            D // 2
        )
        branches.append(local)

    # Global branch
    if use_global:
        global_branch = L.AveragePooling2D(
            pool_size=2
        )(x)

        global_branch = conv_block(
            global_branch,
            D // 2
        )

        global_branch = L.UpSampling2D(
            size=(2, 2)
        )(global_branch)

        branches.append(global_branch)

    # Dilated branch
    if use_dilated:
        dilated = L.Conv2D(
            D // 2,
            3,
            dilation_rate=2,
            padding="same"
        )(x)

        dilated = L.BatchNormalization()(dilated)
        dilated = L.ReLU()(dilated)

        branches.append(dilated)

    # Baseline: no MSTF
    if len(branches) == 0:
        return x_tokens

    fused = L.Concatenate()(branches)

    fused = L.Conv2D(
        D,
        1,
        padding="same"
    )(fused)

    fused_tokens = L.Reshape(
        (H * H, D)
    )(fused)

    return L.Add()([
        fused_tokens,
        x_tokens
    ])


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

def transformer_block(
    x,
    cf,
    use_local=True,
    use_global=True,
    use_dilated=True
):

    # --------------------------------------------------------
    # Multi-Head Self Attention
    # --------------------------------------------------------

    skip = x

    x = L.LayerNormalization()(x)

    x = L.MultiHeadAttention(
        num_heads=cf["num_query_heads"],
        key_dim=cf["head_dim"],
        dropout=cf["dropout_rate"]
    )(x, x)

    x = L.Add()([
        x,
        skip
    ])

    # --------------------------------------------------------
    # MSTF
    # --------------------------------------------------------

    x = MSTF(
        x,
        cf,
        use_local=use_local,
        use_global=use_global,
        use_dilated=use_dilated
    )

    # --------------------------------------------------------
    # Feed Forward Network
    # --------------------------------------------------------

    skip = x

    x = L.LayerNormalization()(x)

    x = L.Dense(
        cf["mlp_dim"],
        activation="gelu"
    )(x)

    x = L.Dropout(
        cf["dropout_rate"]
    )(x)

    x = L.Dense(
        cf["hidden_dim"]
    )(x)

    x = L.Dropout(
        cf["dropout_rate"]
    )(x)

    x = L.Add()([
        x,
        skip
    ])

    return x


# ============================================================
# MODEL
# ============================================================

def build_unetr_mstf(
    cf,
    use_local=True,
    use_global=True,
    use_dilated=True
):

    inputs = L.Input(
        shape=(
            cf["num_patches"],
            cf["patch_size"] ** 2 * 3
        )
    )

    # Patch projection
    x = L.Dense(
        cf["hidden_dim"]
    )(inputs)

    # Positional embedding
    pos_embed = L.Embedding(
        cf["num_patches"],
        cf["hidden_dim"]
    )(
        tf.range(cf["num_patches"])
    )

    x = x + pos_embed

    # --------------------------------------------------------
    # Transformer encoder
    # --------------------------------------------------------

    skips = []

    for i in range(
        cf["num_layers"]
    ):

        x = transformer_block(
            x,
            cf,
            use_local=use_local,
            use_global=use_global,
            use_dilated=use_dilated
        )

        # Layers 3, 6, 9, 12
        if i in [2, 5, 8, 11]:
            skips.append(x)

    s1, s2, s3, s4 = skips

    num_classes = 3

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    def decode(
        decoder_input,
        skip,
        filters
    ):

        decoder_input = L.Conv2DTranspose(
            filters,
            2,
            strides=2,
            padding="same"
        )(decoder_input)

        H_grid = (
            cf["image_size"]
            // cf["patch_size"]
        )

        skip_2d = L.Reshape(
            (
                H_grid,
                H_grid,
                cf["hidden_dim"]
            )
        )(skip)

        target_h = K.int_shape(
            decoder_input
        )[1]

        target_w = K.int_shape(
            decoder_input
        )[2]

        skip_aligned = L.Resizing(
            target_h,
            target_w
        )(skip_2d)

        skip_aligned = conv_block(
            skip_aligned,
            filters
        )

        decoder_input = L.Concatenate()([
            decoder_input,
            skip_aligned
        ])

        decoder_input = conv_block(
            decoder_input,
            filters
        )

        return decoder_input

    # --------------------------------------------------------
    # Decoder neck
    # --------------------------------------------------------

    neck = L.Reshape(
        (
            cf["image_size"] // 16,
            cf["image_size"] // 16,
            cf["hidden_dim"]
        )
    )(s4)

    d1 = decode(
        neck,
        s3,
        512
    )

    d2 = decode(
        d1,
        s2,
        256
    )

    d3 = decode(
        d2,
        s1,
        128
    )

    # --------------------------------------------------------
    # Deep supervision heads
    # --------------------------------------------------------

    out1 = L.Conv2D(
        num_classes,
        1,
        activation="softmax",
        name="deep_sup_1"
    )(d1)

    out1 = L.Resizing(
        cf["image_size"],
        cf["image_size"]
    )(out1)

    out2 = L.Conv2D(
        num_classes,
        1,
        activation="softmax",
        name="deep_sup_2"
    )(d2)

    out2 = L.Resizing(
        cf["image_size"],
        cf["image_size"]
    )(out2)

    out3 = L.Conv2D(
        num_classes,
        1,
        activation="softmax",
        name="deep_sup_3"
    )(d3)

    out3 = L.Resizing(
        cf["image_size"],
        cf["image_size"]
    )(out3)

    # Final prediction
    final_features = L.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(d3)

    final = L.Conv2D(
        num_classes,
        1,
        activation="softmax",
        name="pred_final"
    )(final_features)

    final = L.Resizing(
        cf["image_size"],
        cf["image_size"]
    )(final)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[
            out1,
            out2,
            out3,
            final
        ],
        name="MSTF_UNETR_REFUGE2"
    )


# ============================================================
# METRICS
# ============================================================

def disc_dice_coef(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.squeeze(
        tf.cast(y_true, tf.int32),
        axis=-1
    )

    y_true = tf.one_hot(
        y_true,
        depth=3
    )

    y_true_disc = (
        y_true[..., 1]
        + y_true[..., 2]
    )

    y_pred_disc = (
        y_pred[..., 1]
        + y_pred[..., 2]
    )

    return (
        2.0 * K.sum(
            y_true_disc * y_pred_disc
        )
        + smooth
    ) / (
        K.sum(y_true_disc)
        + K.sum(y_pred_disc)
        + smooth
    )


def cup_dice_coef(y_true, y_pred):

    smooth = 1e-6

    y_true = tf.squeeze(
        tf.cast(y_true, tf.int32),
        axis=-1
    )

    y_true = tf.one_hot(
        y_true,
        depth=3
    )

    y_true_cup = y_true[..., 2]
    y_pred_cup = y_pred[..., 2]

    return (
        2.0 * K.sum(
            y_true_cup * y_pred_cup
        )
        + smooth
    ) / (
        K.sum(y_true_cup)
        + K.sum(y_pred_cup)
        + smooth
    )


def mean_dice_coef(y_true, y_pred):

    return (
        disc_dice_coef(
            y_true,
            y_pred
        )
        +
        cup_dice_coef(
            y_true,
            y_pred
        )
    ) / 2.0


# ============================================================
# LOSS
# ============================================================

def categorical_dice_loss(
    y_true,
    y_pred
):

    y_true = tf.squeeze(
        tf.cast(y_true, tf.int32),
        axis=-1
    )

    y_true_one_hot = tf.one_hot(
        y_true,
        depth=3
    )

    # Class weights from the REFUGE2 notebook
    class_weights = tf.constant(
        [0.2, 2.0, 3.0],
        dtype=tf.float32
    )

    ce_per_class = -tf.reduce_sum(
        y_true_one_hot
        * tf.math.log(
            y_pred + 1e-7
        ),
        axis=-1
    )

    pixel_weights = tf.reduce_sum(
        y_true_one_hot
        * class_weights,
        axis=-1
    )

    ce = tf.reduce_mean(
        ce_per_class
        * pixel_weights
    )

    dice_loss = (
        1.0
        - mean_dice_coef(
            y_true,
            y_pred
        )
    )

    return (
        0.5 * ce
        + 0.5 * dice_loss
    )


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

def get_cosine_scheduler(
    total_epochs,
    warmup_epochs,
    lr_min,
    lr_max
):

    def scheduler(epoch):

        if epoch < warmup_epochs:

            return float(
                lr_min
                +
                (
                    lr_max
                    - lr_min
                )
                * (
                    epoch
                    / warmup_epochs
                )
            )

        progress = (
            epoch - warmup_epochs
        ) / max(
            total_epochs
            - warmup_epochs,
            1
        )

        cosine_decay = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * progress
            )
        )

        return float(
            lr_min
            +
            (
                lr_max
                - lr_min
            )
            * cosine_decay
        )

    return scheduler


# ============================================================
# ONE FOLD
# ============================================================

def train_fold(
    fold,
    train_idx,
    val_idx,
    all_images,
    all_masks,
    preprocess_train,
    preprocess_val
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"STARTING FOLD "
        f"{fold + 1}/{N_SPLITS}"
    )

    print(
        f"{'=' * 60}"
    )

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    train_images = np.array(
        all_images
    )[train_idx]

    train_masks = np.array(
        all_masks
    )[train_idx]

    val_images = np.array(
        all_images
    )[val_idx]

    val_masks = np.array(
        all_masks
    )[val_idx]

    train_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                train_images,
                train_masks
            )
        )
        .shuffle(
            len(train_idx)
        )
        .map(
            preprocess_train,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(
            BATCH_SIZE
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    val_ds = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                val_images,
                val_masks
            )
        )
        .map(
            preprocess_val,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(
            BATCH_SIZE
        )
        .prefetch(
            tf.data.AUTOTUNE
        )
    )

    # --------------------------------------------------------
    # Fresh model for this fold
    # --------------------------------------------------------

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_unetr_mstf(
        CF
    )

    model.compile(

        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LR_MAX,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        ),

        loss={
            "deep_sup_1":
                categorical_dice_loss,

            "deep_sup_2":
                categorical_dice_loss,

            "deep_sup_3":
                categorical_dice_loss,

            "pred_final":
                categorical_dice_loss
        },

        loss_weights={
            "deep_sup_1": 0.1,
            "deep_sup_2": 0.2,
            "deep_sup_3": 0.3,
            "pred_final": 0.4
        },

        metrics={
            "pred_final": [
                disc_dice_coef,
                cup_dice_coef,
                mean_dice_coef
            ]
        }
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"fold_{fold + 1}_best.weights.h5"
    )

    scheduler = get_cosine_scheduler(
        EPOCHS,
        WARMUP_EPOCHS,
        LR_MIN,
        LR_MAX
    )

    callbacks = [

        ModelCheckpoint(
            checkpoint_path,
            monitor=(
                "val_pred_final_"
                "mean_dice_coef"
            ),
            mode="max",
            save_best_only=True,
            save_weights_only=True
        ),

        EarlyStopping(
            monitor=(
                "val_pred_final_"
                "mean_dice_coef"
            ),
            mode="max",
            patience=10,
            restore_best_weights=True
        ),

        LearningRateScheduler(
            scheduler
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    best_dice = max(
        history.history[
            "val_pred_final_mean_dice_coef"
        ]
    )

    print(
        f"\nFold {fold + 1} complete."
    )

    print(
        f"Best validation mean Dice: "
        f"{best_dice:.4f}"
    )

    return {
        "fold": fold + 1,
        "best_val_mean_dice": best_dice,
        "checkpoint": checkpoint_path
    }


# ============================================================
# RUN 5-FOLD CV
# ============================================================

def run_5_fold_cv(
    all_images,
    all_masks,
    preprocess_train,
    preprocess_val
):

    all_images = np.array(
        all_images
    )

    all_masks = np.array(
        all_masks
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        kfold.split(all_images)
    ):

        result = train_fold(
            fold,
            train_idx,
            val_idx,
            all_images,
            all_masks,
            preprocess_train,
            preprocess_val
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    fold_scores = [
        result[
            "best_val_mean_dice"
        ]
        for result in results
    ]

    mean_score = np.mean(
        fold_scores
    )

    best_fold_index = int(
        np.argmax(
            fold_scores
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "5-FOLD CROSS-VALIDATION RESULTS"
    )

    print(
        "=" * 60
    )

    for result in results:

        print(
            f"Fold {result['fold']}: "
            f"{result['best_val_mean_dice']:.4f}"
        )

    print(
        f"\nMean CV Dice: "
        f"{mean_score:.4f}"
    )

    print(
        f"Best Fold: "
        f"{best_fold_index + 1}"
    )

    print(
        f"Best Fold Dice: "
        f"{fold_scores[best_fold_index]:.4f}"
    )

    print(
        "=" * 60
    )

    return results
