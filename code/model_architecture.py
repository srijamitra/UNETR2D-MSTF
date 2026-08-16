"""
UNETR2D-MSTF
Transformer-based Medical Image Segmentation

Model architecture used for ISIC 2018 experiments.

Architecture:
- 2D Vision Transformer (ViT-Base)
- Multi-Scale Token Fusion (MSTF)
- UNETR-style skip connections
- U-Net decoder
- Deep supervision at four scales
- Binary segmentation for ISIC 2018
"""

import tensorflow as tf
from tensorflow.keras import layers as L
import tensorflow.keras.backend as K


def conv_block(x, f):
    """
    Convolutional block used in the MSTF module and decoder.
    """
    x = L.Conv2D(f, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


def MSTF(x_tokens, cf):
    """
    Multi-Scale Token Fusion (MSTF).

    The token sequence is reshaped into a 2D spatial feature map
    and processed through three parallel branches:

    1. Local branch:
       3x3 convolution for local spatial features.

    2. Global-context branch:
       Average pooling followed by convolution and upsampling.

    3. Dilated branch:
       3x3 dilated convolution with dilation rate 2.

    The three representations are concatenated, projected back
    to the original embedding dimension, reshaped into tokens,
    and added through a residual connection.
    """

    H = cf["image_size"] // cf["patch_size"]
    D = cf["hidden_dim"]

    # Convert tokens back to a spatial feature map
    x = L.Reshape((H, H, D))(x_tokens)

    # --------------------------------------------------------
    # Branch 1: Local context
    # --------------------------------------------------------
    s1 = conv_block(x, D // 2)

    # --------------------------------------------------------
    # Branch 2: Global context
    # --------------------------------------------------------
    s2 = L.AveragePooling2D(pool_size=2)(x)
    s2 = conv_block(s2, D // 2)
    s2 = L.UpSampling2D(size=(2, 2))(s2)

    # --------------------------------------------------------
    # Branch 3: Dilated / wider context
    # --------------------------------------------------------
    s3 = L.Conv2D(
        D // 2,
        3,
        dilation_rate=2,
        padding="same"
    )(x)
    s3 = L.BatchNormalization()(s3)
    s3 = L.ReLU()(s3)

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------
    fused = L.Concatenate()([s1, s2, s3])
    fused = L.Conv2D(D, 1, padding="same")(fused)

    # Convert back to token representation
    fused_tokens = L.Reshape((H * H, D))(fused)

    # Residual connection
    return L.Add()([fused_tokens, x_tokens])


def transformer_block(x, cf):
    """
    Transformer encoder block with:

    Layer Normalization
        -> Multi-Head Self-Attention
        -> Residual connection
        -> MSTF
        -> Layer Normalization
        -> MLP
        -> Residual connection
    """

    # --------------------------------------------------------
    # Multi-Head Self-Attention
    # --------------------------------------------------------
    skip = x

    x = L.LayerNormalization()(x)

    x = L.MultiHeadAttention(
        num_heads=cf["num_query_heads"],
        key_dim=cf["head_dim"],
        dropout=cf["dropout_rate"]
    )(x, x)

    x = L.Add()([x, skip])

    # --------------------------------------------------------
    # Multi-Scale Token Fusion
    # --------------------------------------------------------
    x = MSTF(x, cf)

    # --------------------------------------------------------
    # MLP / Feed Forward Network
    # --------------------------------------------------------
    skip = x

    x = L.LayerNormalization()(x)

    x = L.Dense(
        cf["mlp_dim"],
        activation="gelu"
    )(x)

    x = L.Dropout(cf["dropout_rate"])(x)

    x = L.Dense(cf["hidden_dim"])(x)

    x = L.Dropout(cf["dropout_rate"])(x)

    x = L.Add()([x, skip])

    return x


def build_unetr_mstf(cf):
    """
    Build the UNETR2D-MSTF model for ISIC 2018.

    The architecture uses:
    - 16x16 image patches
    - 256 tokens
    - 512-dimensional token embeddings
    - 12 transformer layers
    - Skip connections from layers 3, 6, 9 and 12
    - Four deep-supervision prediction heads
    - Binary segmentation output
    """

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------
    inputs = L.Input(
        (
            cf["num_patches"],
            cf["patch_size"] ** 2 * 3
        )
    )

    # --------------------------------------------------------
    # Linear projection + positional embedding
    # --------------------------------------------------------
    x = L.Dense(cf["hidden_dim"])(inputs)

    pos_embed = L.Embedding(
        cf["num_patches"],
        cf["hidden_dim"]
    )(tf.range(cf["num_patches"]))

    x = x + pos_embed

    # --------------------------------------------------------
    # Transformer encoder
    # --------------------------------------------------------
    skips = []

    for i in range(cf["num_layers"]):

        x = transformer_block(x, cf)

        # UNETR skip connections:
        # layer 3, 6, 9 and 12
        if i in [2, 5, 8, 11]:
            skips.append(x)

    s1, s2, s3, s4 = skips

    # ISIC 2018:
    # class 0 = background
    # class 1 = lesion
    num_classes = cf["num_classes"]

    # --------------------------------------------------------
    # Decoder block
    # --------------------------------------------------------
    def decode(x, skip, f, name=None):

        # Upsample decoder representation
        x = L.Conv2DTranspose(
            f,
            2,
            strides=2,
            padding="same"
        )(x)

        # Convert transformer tokens to spatial representation
        H_grid = cf["image_size"] // cf["patch_size"]

        skip_2d = L.Reshape(
            (
                H_grid,
                H_grid,
                cf["hidden_dim"]
            )
        )(skip)

        # Align spatial dimensions
        target_h = K.int_shape(x)[1]
        target_w = K.int_shape(x)[2]

        skip_aligned = L.Resizing(
            target_h,
            target_w
        )(skip_2d)

        # Skip connection
        x = L.Concatenate()([
            x,
            conv_block(skip_aligned, f)
        ])

        x = conv_block(x, f)

        # Deep-supervision prediction head
        if name:

            out = L.Conv2D(
                num_classes,
                1,
                activation="softmax",
                name=name
            )(x)

            return L.Resizing(
                cf["image_size"],
                cf["image_size"]
            )(out), x

        return x

    # --------------------------------------------------------
    # Bottleneck
    # --------------------------------------------------------
    neck = L.Reshape(
        (
            cf["image_size"] // 16,
            cf["image_size"] // 16,
            cf["hidden_dim"]
        )
    )(s4)

    # --------------------------------------------------------
    # Decoder with deep supervision
    # --------------------------------------------------------

    # 16 -> 32
    out1, x = decode(
        neck,
        s3,
        512,
        "pred_1"
    )

    # 32 -> 64
    out2, x = decode(
        x,
        s2,
        256,
        "pred_2"
    )

    # 64 -> 128
    out3, x = decode(
        x,
        s1,
        128,
        "pred_3"
    )

    # --------------------------------------------------------
    # Final decoder stage: 128 -> 256
    # --------------------------------------------------------
    x = L.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(x)

    final = L.Conv2D(
        num_classes,
        1,
        activation="softmax",
        name="pred_final"
    )(x)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = tf.keras.Model(
        inputs,
        [
            out1,
            out2,
            out3,
            final
        ],
        name="MSTF_UNETR_ISIC2018_Binary"
    )

    return model
