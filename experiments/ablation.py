"""
UNETR2D-MSTF
Ablation Study

Configurations:
    1. Baseline UNETR       -> No MSTF
    2. MSTF Local Only
    3. MSTF Global Only
    4. MSTF Dilated Only
    5. Full MSTF

The paper evaluates these configurations on:
    - REFUGE2
    - ISIC 2018
"""

import os
import gc
import json
import numpy as np
import tensorflow as tf

from train import (
    CF,
    build_unetr_mstf,
    categorical_dice_loss,
    disc_dice_coef,
    cup_dice_coef,
    mean_dice_coef,
    BATCH_SIZE,
    EPOCHS,
    LR_MAX,
    LR_MIN,
    WARMUP_EPOCHS,
    get_cosine_scheduler,
)


# ============================================================
# CONFIGURATION
# ============================================================

ABLATION_CONFIGS = {

    "baseline_no_mstf": {
        "use_local": False,
        "use_global": False,
        "use_dilated": False,
    },

    "local_only": {
        "use_local": True,
        "use_global": False,
        "use_dilated": False,
    },

    "global_only": {
        "use_local": False,
        "use_global": True,
        "use_dilated": False,
    },

    "dilated_only": {
        "use_local": False,
        "use_global": False,
        "use_dilated": True,
    },

    "full_mstf": {
        "use_local": True,
        "use_global": True,
        "use_dilated": True,
    },
}


RESULTS_DIR = "results/ablation"
CHECKPOINT_DIR = "checkpoints/ablation"


# ============================================================
# MODEL COMPILATION
# ============================================================

def compile_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LR_MAX,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),

        loss={
            "deep_sup_1": categorical_dice_loss,
            "deep_sup_2": categorical_dice_loss,
            "deep_sup_3": categorical_dice_loss,
            "pred_final": categorical_dice_loss,
        },

        loss_weights={
            "deep_sup_1": 0.1,
            "deep_sup_2": 0.2,
            "deep_sup_3": 0.3,
            "pred_final": 0.4,
        },

        metrics={
            "pred_final": [
                disc_dice_coef,
                cup_dice_coef,
                mean_dice_coef,
            ]
        },
    )

    return model


# ============================================================
# TRAIN ONE ABLATION CONFIGURATION
# ============================================================

def train_ablation_config(
    config_name,
    config,
    train_ds,
    val_ds,
):

    print("\n" + "=" * 70)

    print(
        f"ABLATION: {config_name}"
    )

    print("=" * 70)

    print(
        f"Local branch   : "
        f"{config['use_local']}"
    )

    print(
        f"Global branch  : "
        f"{config['use_global']}"
    )

    print(
        f"Dilated branch : "
        f"{config['use_dilated']}"
    )

    # --------------------------------------------------------
    # Clear previous model
    # --------------------------------------------------------

    tf.keras.backend.clear_session()
    gc.collect()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_unetr_mstf(

        CF,

        use_local=config["use_local"],
        use_global=config["use_global"],
        use_dilated=config["use_dilated"],
    )

    model = compile_model(
        model
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
        f"{config_name}.weights.h5"
    )

    # --------------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------------

    scheduler = get_cosine_scheduler(
        EPOCHS,
        WARMUP_EPOCHS,
        LR_MIN,
        LR_MAX,
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            checkpoint_path,

            monitor=(
                "val_pred_final_"
                "mean_dice_coef"
            ),

            mode="max",

            save_best_only=True,

            save_weights_only=True,
        ),

        tf.keras.callbacks.EarlyStopping(

            monitor=(
                "val_pred_final_"
                "mean_dice_coef"
            ),

            mode="max",

            patience=10,

            restore_best_weights=True,
        ),

        tf.keras.callbacks.LearningRateScheduler(
            scheduler
        ),
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1,
    )

    # --------------------------------------------------------
    # Best validation score
    # --------------------------------------------------------

    best_epoch = int(
        np.argmax(
            history.history[
                "val_pred_final_mean_dice_coef"
            ]
        )
    )

    best_mean_dice = float(
        history.history[
            "val_pred_final_mean_dice_coef"
        ][best_epoch]
    )

    best_disc_dice = float(
        history.history[
            "val_pred_final_disc_dice_coef"
        ][best_epoch]
    )

    best_cup_dice = float(
        history.history[
            "val_pred_final_cup_dice_coef"
        ][best_epoch]
    )

    result = {

        "configuration": config_name,

        "local_branch":
            config["use_local"],

        "global_branch":
            config["use_global"],

        "dilated_branch":
            config["use_dilated"],

        "best_epoch":
            best_epoch + 1,

        "mean_dice":
            best_mean_dice,

        "disc_dice":
            best_disc_dice,

        "cup_dice":
            best_cup_dice,

        "checkpoint":
            checkpoint_path,
    }

    print(
        f"\n{config_name}"
    )

    print(
        f"Mean Dice : "
        f"{best_mean_dice:.4f}"
    )

    print(
        f"Disc Dice : "
        f"{best_disc_dice:.4f}"
    )

    print(
        f"Cup Dice  : "
        f"{best_cup_dice:.4f}"
    )

    return result


# ============================================================
# RUN COMPLETE ABLATION
# ============================================================

def run_ablation(
    train_ds,
    val_ds,
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results = []

    # --------------------------------------------------------
    # Run all five configurations
    # --------------------------------------------------------

    for config_name, config in (
        ABLATION_CONFIGS.items()
    ):

        result = train_ablation_config(

            config_name,

            config,

            train_ds,

            val_ds,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    result_path = os.path.join(
        RESULTS_DIR,
        "ablation_results.json"
    )

    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print(
        f"{'Configuration':<25}"
        f"{'Mean Dice':>12}"
        f"{'Disc Dice':>12}"
        f"{'Cup Dice':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['configuration']:<25}"
            f"{result['mean_dice']:>12.4f}"
            f"{result['disc_dice']:>12.4f}"
            f"{result['cup_dice']:>12.4f}"
        )

    print("=" * 70)

    return results


# ============================================================
# EXPECTED PAPER TABLE
# ============================================================

PAPER_RESULTS = {

    "baseline_no_mstf": {
        "refuge_mean_dice": 85.43,
        "refuge_disc_dice": 86.37,
        "refuge_cup_dice": 84.50,
        "isic_mean_dice": 85.40,
    },

    "local_only": {
        "refuge_mean_dice": 86.53,
        "refuge_disc_dice": 87.36,
        "refuge_cup_dice": 85.69,
        "isic_mean_dice": 86.50,
    },

    "global_only": {
        "refuge_mean_dice": 85.12,
        "refuge_disc_dice": 86.31,
        "refuge_cup_dice": 83.92,
        "isic_mean_dice": 86.70,
    },

    "dilated_only": {
        "refuge_mean_dice": 85.83,
        "refuge_disc_dice": 86.80,
        "refuge_cup_dice": 84.86,
        "isic_mean_dice": 86.10,
    },

    "full_mstf": {
        "refuge_mean_dice": 87.20,
        "refuge_disc_dice": 88.20,
        "refuge_cup_dice": 86.10,
        "isic_mean_dice": 87.50,
    },
}


def print_paper_results():

    print("\n")
    print("=" * 70)
    print("RESULTS REPORTED IN PAPER")
    print("=" * 70)

    for name, values in PAPER_RESULTS.items():

        print(
            f"\n{name}"
        )

        print(
            f"REFUGE2 Mean Dice : "
            f"{values['refuge_mean_dice']:.2f}"
        )

        print(
            f"REFUGE2 Disc Dice : "
            f"{values['refuge_disc_dice']:.2f}"
        )

        print(
            f"REFUGE2 Cup Dice  : "
            f"{values['refuge_cup_dice']:.2f}"
        )

        print(
            f"ISIC Mean Dice    : "
            f"{values['isic_mean_dice']:.2f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_paper_results()

    print(
        "\n"
        "To actually run the ablation, import "
        "run_ablation() from the training pipeline."
    )
