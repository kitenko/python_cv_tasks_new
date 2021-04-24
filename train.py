import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

import data_generator, cnn_model
from config import NUMBER_OF_CLASSES, LOGS_DIR, INPUT_SHAPE, LEARNING_RATE, EPOCHS, JSON_FILE_PATH


def train(dataset_path_json: str, save_path: str) -> None:
    """
    Training colour classifier

    :param dataset_path_json: path to json file.
    :param save_path: path to save weights and training logs.
    """
    log_dir = os.path.join(save_path)
    os.makedirs(log_dir, exist_ok=True)

    train_data_gen = data_generator.DataGenerator(dataset_path_json, is_train=True)
    test_data_gen = data_generator.DataGenerator(dataset_path_json, is_train=False)

    model = cnn_model.My_Mega_Model(input_shape=INPUT_SHAPE, num_classes=NUMBER_OF_CLASSES, activation_type='Mish',
                                    input_name='Model', regularization=0.0005).build()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  metrics=['accuracy'])
    model.summary()
    early = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(log_dir, 'model.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                        validation_steps=len(test_data_gen), epochs=EPOCHS,
                        callbacks=[model_checkpoint_callback, early])

fit = train(JSON_FILE_PATH, LOGS_DIR)
