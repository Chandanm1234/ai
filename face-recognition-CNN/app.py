import subprocess
import sys

# Function to install required packages
def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Install required packages
install_packages()

import os
import shutil
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import Policy, set_global_policy
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

app = Flask(__name__)

# Ensure directories exist
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Paths
MODEL_PATH = 'C:/face-recognition-CNN/pretrained_model/pretrained_model.h5'
FACE_CASCADE_PATH = 'C:/face-recognition-CNN/haarcascade_frontalface_default.xml'
RF_MODEL_PATH = 'C:/face-recognition-CNN/static/model/random_forest/face_recognition_rf_model.pkl'
CNN_MODEL_PATH = 'C:/face-recognition-CNN/static/model/cnn/face_recognition_cnn_model.h5'
CNN_CONFUSION_MATRIX_PATH = 'C:/face-recognition-CNN/static/model/cnn/confusion_matrix.png'
RF_CONFUSION_MATRIX_PATH = 'C:/face-recognition-CNN/static/model/random_forest/confusion_matrix.png'
ATTENDANCE_LOG_PATH = 'C:/face-recognition-CNN/Attendance/attendance_log.csv'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Enable mixed precision
policy = Policy('mixed_float16')
set_global_policy(policy)

# Set memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Function to capture face images
def capture_face_images(username, userid, nimgs=10):
    userimagefolder = f'static/faces/{username}_{userid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cam = cv2.VideoCapture(0)
    count = 0
    while count < nimgs:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            cv2.imwrite(f'{userimagefolder}/{username}_{userid}_{count}.jpg', face)
            count += 1
            if count >= nimgs:
                break

    cam.release()

# Function to train CNN model
def train_cnn_model():
    data_dir = 'static/faces'
    output_cnn_model_path = CNN_MODEL_PATH
    output_static_dir = 'static/model/cnn'

    with tf.device('/CPU:0'):
        datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

        train_generator = datagen.flow_from_directory(
            data_dir, 
            target_size=(224, 224), 
            batch_size=8,  # Reduced batch size
            class_mode='categorical', 
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            data_dir, 
            target_size=(224, 224), 
            batch_size=8,  # Reduced batch size
            class_mode='categorical', 
            subset='validation'
        )

    num_classes = train_generator.num_classes

    with tf.device('/GPU:0'):
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:150]:  # Freeze more layers
            layer.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer='l2'),
            BatchNormalization(),
            Dropout(0.5),  # Adjusted dropout rate
            Dense(num_classes, activation='softmax', dtype='float32')  # Specify dtype for mixed precision
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(output_cnn_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

        history = model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[checkpoint, early_stopping, reduce_lr])

    with tf.device('/CPU:0'):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.savefig(os.path.join(output_static_dir, 'performance.png'))

        y_pred = model.predict(validation_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = validation_generator.classes
        class_labels = list(validation_generator.class_indices.keys())

        cm = confusion_matrix(y_true, y_pred_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(CNN_CONFUSION_MATRIX_PATH)

    print(f'CNN Model Accuracy: {accuracy_score(y_true, y_pred_classes):.2f}')

# Function to train Random Forest model
def train_rf_model():
    data_dir = 'static/faces'
    output_rf_model_path = RF_MODEL_PATH
    output_static_dir = 'static/model/random_forest'

    with tf.device('/CPU:0'):
        datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

        train_generator = datagen.flow_from_directory(
            data_dir, 
            target_size=(224, 224), 
            batch_size=8,  # Reduced batch size
            class_mode='categorical', 
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            data_dir, 
            target_size=(224, 224), 
            batch_size=8,  # Reduced batch size
            class_mode='categorical', 
            subset='validation'
        )

        x_train, y_train = [], []
        for i in range(len(train_generator)):
            x, y = train_generator[i]
            x_train.extend(x)
            y_train.extend(y)

        x_train = np.array(x_train)
        y_train = np.argmax(y_train, axis=1)

    with tf.device('/GPU:0'):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x_train.reshape(x_train.shape[0], -1), y_train)
        joblib.dump(rf, output_rf_model_path)

    with tf.device('/CPU:0'):
        y_pred = rf.predict(validation_generator[0][0].reshape(validation_generator[0][0].shape[0], -1))
        y_true = np.argmax(validation_generator[0][1], axis=1)

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(RF_CONFUSION_MATRIX_PATH)

    print(f'Random Forest Model Accuracy: {accuracy_score(y_true, y_pred):.2f}')

# Function to log attendance to CSV
def log_attendance_to_csv(records):
    if not os.path.exists(ATTENDANCE_LOG_PATH):
        df = pd.DataFrame(columns=['s.no', 'name', 'id', 'time', 'date'])
        df.to_csv(ATTENDANCE_LOG_PATH, index=False)

    df = pd.read_csv(ATTENDANCE_LOG_PATH)
    start_index = len(df)
    new_records = []
    for i, record in enumerate(records):
        new_records.append({
            's.no': start_index + i + 1,
            'name': record['name'],
            'id': record['id'],
            'time': record['time'],
            'date': record['date']
        })
    new_df = pd.DataFrame(new_records)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(ATTENDANCE_LOG_PATH, index=False)

# Routes
@app.route('/')
def home():
    faces_dir = 'static/faces'
    user_count = len([name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))])
    return render_template('home.html', totalreg=user_count)

@app.route('/listusers')
def list_users():
    users = []
    faces_dir = 'static/faces'
    for folder in os.listdir(faces_dir):
        if os.path.isdir(os.path.join(faces_dir, folder)):
            user_info = folder.split('_')
            if len(user_info) == 2:
                users.append({'name': user_info[0], 'id': user_info[1]})
    return render_template('listusers.html', userlist=users)

@app.route('/attendance')
def attendance():
    records = []
    if os.path.exists(ATTENDANCE_LOG_PATH):
        df = pd.read_csv(ATTENDANCE_LOG_PATH)
        records = df.to_dict('records')
    return render_template('attendance.html', records=records)

@app.route('/getimages', methods=['POST'])
def get_images():
    username = request.form['newusername']
    userid = request.form['newuserid']
    capture_face_images(username, userid)
    return jsonify({'status': 'success'})

@app.route('/train', methods=['POST'])
def train():
    try:
        train_cnn_model()
        return jsonify({'message': 'Training complete'})
    except Exception as e:
        return jsonify({'message': 'An error occurred during training', 'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        train_rf_model()
        return jsonify({'message': 'Retraining complete'})
    except Exception as e:
        return jsonify({'message': 'An error occurred during retraining', 'error': str(e)}), 500

@app.route('/deleteuser', methods=['GET'])
def delete_user():
    user = request.args.get('user')
    user_path = os.path.join('static/faces', user)
    if os.path.exists(user_path):
        shutil.rmtree(user_path)
        return redirect(url_for('list_users'))
    return 'User not found', 404

@app.route('/getimages', methods=['GET'])
def show_images():
    user = request.args.get('user')
    user_path = os.path.join('static/faces', user)
    if os.path.exists(user_path):
        images = [url_for('static', filename=f'faces/{user}/{img}') for img in os.listdir(user_path)]
        return jsonify({'images': images})
    return jsonify({'images': []})

@app.route('/takeattendance', methods=['POST'])
def take_attendance():
    with tf.device('/CPU:0'):
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()
        if not ret:
            return jsonify({'message': 'Failed to capture image'})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

        attendance_record = []

    with tf.device('/GPU:0'):
        model = load_model(CNN_MODEL_PATH)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = np.expand_dims(face, axis=0) / 255.0
            prediction = model.predict(face)
            label = np.argmax(prediction, axis=1)[0]

            user_info = get_user_info_from_label(label)
            if user_info:
                name, userid = user_info
                time = datetime.now().strftime("%H:%M:%S")
                date = datetime.now().strftime("%Y-%m-%d")
                attendance_record.append({'name': name, 'id': userid, 'time': time, 'date': date})
    
    log_attendance_to_csv(attendance_record)
    return jsonify({'message': 'Attendance taken', 'attendance': attendance_record})

@app.route('/attendancesummary')
def attendance_summary():
    total_days = request.args.get('total_days')
    if total_days is None:
        return "Total number of days not provided", 400

    try:
        total_days = int(total_days)
    except ValueError:
        return "Total number of days must be an integer", 400

    summary = []
    if os.path.exists(ATTENDANCE_LOG_PATH):
        df = pd.read_csv(ATTENDANCE_LOG_PATH)
        summary_df = df.groupby(['name', 'id']).size().reset_index(name='days')
        summary_df['percentage'] = (summary_df['days'] / total_days) * 100
        summary = summary_df.to_dict('records')
    return render_template('attendance_summary.html', summary=summary)

def get_user_info_from_label(label):
    faces_dir = 'static/faces'
    user_dirs = os.listdir(faces_dir)
    for user_dir in user_dirs:
        if str(label) in user_dir:
            name, userid = user_dir.split('_')
            return name, userid
    return None

if __name__ == '__main__':
    app.run(debug=True)
