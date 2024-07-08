import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import math

# Paths
train_dirs = [
    'C:\\Users\\chand\\OneDrive\\Desktop\\train',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train2',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train3',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train4',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train5',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train6',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train7',
    'C:\\Users\\chand\\OneDrive\\Desktop\\train8'
]
val_dir = 'C:\\Users\\chand\\OneDrive\\Desktop\\val'
pretrained_model_path = 'C:\\face-recognition-CNN\\pretrained_model'

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Advanced Image Data Generator with CutMix
def cutmix(batch, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = tf.random.shuffle(tf.range(batch.shape[0]))
    x1 = batch
    x2 = tf.gather(batch, rand_index)
    bbx1, bby1, bbx2, bby2 = get_bbox(batch.shape[1], batch.shape[2], lam)
    new_batch = x1 * (1 - lam) + x2 * lam
    return new_batch

def get_bbox(width, height, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2

with tf.device('/CPU:0'):
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical'
    )

# Load InceptionResNetV2 with pretrained weights
with tf.device('/GPU:0'):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the initial layers of the base model
    for layer in base_model.layers[:80]:
        layer.trainable = False

    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.7),
        Dense(60, activation='softmax')  # Adjust to 60 units for 60 classes
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks to save the best model, early stopping, and learning rate scheduler
    checkpoint_cb = ModelCheckpoint(
        pretrained_model_path,
        save_best_only=True
    )

    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

    def scheduler(epoch, lr):
        return lr * math.exp(-0.1)

    lr_scheduler_cb = LearningRateScheduler(scheduler)

# Initialize lists to store the history of accuracy and loss
train_accuracy_history = []
val_accuracy_history = []
train_loss_history = []
val_loss_history = []

# Train the model iteratively using each train directory
for train_dir in train_dirs:
    with tf.device('/CPU:0'):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.3,
            zoom_range=0.3,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            preprocessing_function=cutmix
        )

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )

    # Train the model with partially unfrozen base model
    with tf.device('/GPU:0'):
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=10,
            validation_data=val_generator,
            validation_steps=val_generator.samples // val_generator.batch_size,
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, lr_scheduler_cb]
        )

    # Append the history
    train_accuracy_history.extend(history.history['accuracy'])
    val_accuracy_history.extend(history.history['val_accuracy'])
    train_loss_history.extend(history.history['loss'])
    val_loss_history.extend(history.history['val_loss'])

# Save the final model
with tf.device('/CPU:0'):
    model.save(pretrained_model_path)

# Plotting combined accuracy and loss
with tf.device('/CPU:0'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_history, label='Train Accuracy')
    plt.plot(val_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('C:\\face-recognition-CNN\\pretrained_model\\combined_accuracy_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('C:\\face-recognition-CNN\\pretrained_model\\combined_loss_plot.png')

# Evaluate the model on the training dataset
with tf.device('/GPU:0'):
    for train_dir in train_dirs:
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical'
        )
        train_loss, train_accuracy = model.evaluate(train_generator, steps=train_generator.samples // train_generator.batch_size)
        print(f'Training loss for {train_dir}: {train_loss:.4f}')
        print(f'Training accuracy for {train_dir}: {train_accuracy:.4f}')
