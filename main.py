import os
import tensorflow as tf
from tensorflow import keras
import neptune.new as neptune
from tensorflow.keras import layers
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from config import build_parser

# Select project
run = neptune.init(project='team-menagerie/testing',
                   tags=['with filter'],
                   api_token=os.environ['NEPTUNE_API_TOKEN'])

# data

parser = build_parser()
args = vars(parser.parse_args())
run['parameters'] = args

num_skipped = 0
for folder_path in ("/mnt/storage/data/stella/PetImages/Cat", "/mnt/storage/data/stella/PetImages/Dog"):
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/storage/data/stella/PetImages",
    validation_split=0.2,
    subset="training",
    seed=args['seed'],
    image_size=args['image_size'],
    batch_size=args['batch_size'],
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/mnt/storage/data/stella/PetImages",
    validation_split=0.2,
    subset="validation",
    seed=args['seed'],
    image_size=args['image_size'],
    batch_size=args['batch_size'],
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(args['rotation_factor']),
    ]
)


# model

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(args['dropout'])(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=args['image_size'] + (3,), num_classes=2)

# train

epochs = 1

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=args['patience']),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
neptune_cbks = NeptuneCallback(run=run, base_namespace='metrics')

model.compile(
    optimizer=keras.optimizers.Adam(args['learning_rate']),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=neptune_cbks, validation_data=val_ds,
)
