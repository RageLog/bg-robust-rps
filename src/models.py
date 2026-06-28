"""
models.py - Multi-Architecture Model Builder
=============================================
Builds transfer-learning models with proper per-backbone preprocessing,
gradient clipping, and a two-phase freeze/unfreeze strategy to prevent
gradient explosion under mixed_float16.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# ---- Per-backbone preprocessing layers ----
# Each Keras Application expects a specific pixel format.
# Our data_loader delivers [0, 1] floats. The Lambda below
# scales them back to [0, 255], then each branch applies
# the official preprocess_input (caffe / torch / tf mode).

_PREPROCESS_FN = {
    "ResNet50":        tf.keras.applications.resnet50.preprocess_input,
    "VGG16":           tf.keras.applications.vgg16.preprocess_input,
    "DenseNet121":     tf.keras.applications.densenet.preprocess_input,
}

# How many layers to FREEZE from the bottom of each backbone
# during the initial warm-up phase.  None = freeze ALL.
_FREEZE_STRATEGY = {
    "EfficientNetV2B0": None,   # freezes entire backbone initially
    "ResNet50":         None,
    "MobileNetV3Small": None,
    "DenseNet121":      None,
    "VGG16":            None,
}

WARMUP_EPOCHS = 3   # epochs to train only the head before unfreezing


def build_model(model_name="EfficientNetV2B0", compile_model=True, head_strategy="standard"):
    print(f"\n[INFO] Building Model: {model_name} | Head Strategy: {head_strategy}")

    inputs = layers.Input(shape=config.IMG_SHAPE)

    # Scale [0,1] -> [0,255] for Keras Applications
    x_scaled = layers.Rescaling(scale=255.0, name="rescale_to_255")(inputs)

    # ---- Backbone Selection ----
    if model_name == "EfficientNetV2B0":
        # EfficientNetV2 has its own built-in preprocessing
        base_model = tf.keras.applications.EfficientNetV2B0(
            input_tensor=x_scaled,
            include_top=False,
            weights='imagenet',
            include_preprocessing=True
        )
    elif model_name == "MobileNetV3Small":
        base_model = tf.keras.applications.MobileNetV3Small(
            input_tensor=x_scaled,
            include_top=False,
            weights='imagenet',
            include_preprocessing=True
        )
    elif model_name in _PREPROCESS_FN:
        # ResNet50, VGG16, DenseNet121: apply their official preprocessing
        x_pre = layers.Lambda(
            _PREPROCESS_FN[model_name], name=f"preprocess_{model_name.lower()}"
        )(x_scaled)

        backbone_cls = {
            "ResNet50":    tf.keras.applications.ResNet50,
            "VGG16":       tf.keras.applications.VGG16,
            "DenseNet121": tf.keras.applications.DenseNet121,
        }
        base_model = backbone_cls[model_name](
            input_tensor=x_pre,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # ---- Phase 1 Strategy: Freeze backbone, train head only ----
    # This prevents gradient explosion during early epochs.
    base_model.trainable = False
    print(f"[INFO] Backbone frozen for {WARMUP_EPOCHS}-epoch warm-up phase.")

    # Keep BatchNorm layers always in inference mode for transfer learning
    x = base_model.output

    # ---- Classification Head ----
    if head_strategy == "spatial_pooling":
        # Combines Average (overall presence) and Max (peak presence) pooling
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg_pool, max_pool])
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    elif head_strategy == "attention":
        # Squeeze-and-Excitation (SE) Block for Channel Attention before GAP
        channels = x.shape[-1]
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(channels // 8, activation='relu')(se)
        se = layers.Dense(channels, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, channels))(se)
        x = layers.Multiply()([x, se])
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    else:
        # Standard GAP
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Force float32 output for numerical stability with mixed precision
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax',
                           dtype='float32')(x)

    model = models.Model(inputs, outputs, name=f"RPS_{model_name}")

    if compile_model:
        # Gradient clipping prevents NaN under mixed_float16
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    return model


def unfreeze_model(model, model_name="EfficientNetV2B0"):
    """Unfreezes the backbone for fine-tuning (Phase 2).
    Called after the initial warm-up epochs complete.
    Keeps BatchNorm layers frozen to preserve pretrained statistics.

    Note: When using `input_tensor=`, Keras flattens the backbone layers
    directly into the parent model, so there is no nested sub-model to find.
    We unfreeze all layers except BatchNorm and our custom head layers.
    """
    # Names of head layers we added (these are already trainable)
    head_layer_names = {
        "rescale_to_255",
        f"preprocess_{model_name.lower()}",
    }

    unfrozen = 0
    frozen_bn = 0

    for layer in model.layers:
        # Skip input, rescaling, and preprocessing layers
        if layer.name in head_layer_names:
            continue
        if isinstance(layer, layers.InputLayer):
            continue

        if isinstance(layer, layers.BatchNormalization):
            # Keep ALL BatchNorm layers frozen (pretrained statistics)
            layer.trainable = False
            frozen_bn += 1
        else:
            # Unfreeze everything else
            layer.trainable = True
            unfrozen += 1

    # Re-compile with a lower learning rate for fine-tuning
    fine_tune_lr = config.LEARNING_RATE / 10.0
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=fine_tune_lr,
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"[INFO] Backbone UNFROZEN for fine-tuning. "
          f"Unfrozen layers: {unfrozen}, Frozen BN: {frozen_bn}, "
          f"Fine-tune LR: {fine_tune_lr:.1e}")

    return model

def unfreeze_model_progressive(model, model_name="EfficientNetV2B0", stage=1):
    """
    Progressive unfreezing strategy with discriminative learning rates:
    stage=1: Unfreeze top 50% of the backbone.
    stage=2: Unfreeze 100% of the backbone.
    Always keeps BatchNormalization layers frozen.
    """
    head_layer_names = {
        "rescale_to_255",
        f"preprocess_{model_name.lower()}",
    }

    # Identify backbone layers (everything before our GlobalAveragePooling2D head)
    backbone_layers = []
    for layer in model.layers:
        if layer.name in head_layer_names or isinstance(layer, layers.InputLayer):
            continue
        if isinstance(layer, layers.GlobalAveragePooling2D):
            break
        backbone_layers.append(layer)

    total_backbone_layers = len(backbone_layers)

    if stage == 1:
        # Phase 2: Unfreeze top 50%
        unfreeze_start_idx = total_backbone_layers // 2
        lr = config.LEARNING_RATE / 10.0  # Descending LR step 1
    else:
        # Phase 3: Unfreeze all 100%
        unfreeze_start_idx = 0
        lr = config.LEARNING_RATE / 50.0  # Descending LR step 2

    unfrozen = 0
    frozen_bn = 0

    # First freeze the entire backbone to reset appropriately
    for layer in backbone_layers:
        layer.trainable = False

    # Then selectively unfreeze the target portion (top X%)
    for layer in backbone_layers[unfreeze_start_idx:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
        else:
            layer.trainable = True
            unfrozen += 1

    # Re-compile with the new discriminative learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"[INFO] Progressive Unfreeze Stage {stage}: Unfrozen {unfrozen} layers (top portion). Frozen BN: {frozen_bn} | Discriminative LR: {lr:.1e}")

    return model