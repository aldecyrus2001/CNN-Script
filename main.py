import base64
import os


import requests

from datetime import datetime
import PIL
from matplotlib import pyplot as plt
import cv2
import shutil
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from flask_cors import CORS  # Import the CORS extension
from werkzeug.utils import secure_filename
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet152, ResNet50, ResNet101
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import subprocess
# from flask_ngrok3 import run_with_ngrok


# Load class names from the file


app = Flask(__name__)
CORS(app)

allowed_image_extensions = {'png', 'jpg', 'jpeg'}

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if request.method == 'POST':

        # data = request.get_json()  # Parse JSON data from the request body
        # classification = request.files['livestockClassification']
        classification = request.form['livestockClassification']
        file = request.files['imagefile']

        if(classification == "Swine"):
            model = tf.keras.models.load_model('./model/Swine/best_model.h5')

            with open('./model/Swine/class_names.txt', 'r') as files:
                class_names = [line.strip() for line in files]
        elif (classification == "Cattle"):
            model = tf.keras.models.load_model('./model/Cattle/best_model.h5')

            with open('./model/Cattle/class_names.txt', 'r') as files:
                class_names = [line.strip() for line in files]

        elif (classification == "Goat"):
            model = tf.keras.models.load_model('./model/Goat/best_model.h5')

            with open('./model/Goat/class_names.txt', 'r') as files:
                class_names = [line.strip() for line in files]

        if file:
            img = Image.open(file)
            # Resize the image to (150, 150) pixels (same as in Colab)
            img = img.convert("RGB")
            img = img.resize((150, 150))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            prediction = model.predict(img)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_label = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])

            # Convert confidence to percentage
            confidence_percent = round(confidence * 100, 2)

            print("Raw predictions:", prediction)
            print("Predictions:", predicted_class_label)
            print("Confidence:", confidence_percent, "%")

            # Create a JSON response
            response_data = {
                # 'prediction': prediction,
                'predicted_class': predicted_class_label,
                'confidence': confidence_percent
            }
            return jsonify(response_data)
        else:
            return "Failed to process the image file", 400



@app.route("/upload", methods=["POST"])
def upload():
    if request.method == 'POST':
        # Get the disease name
        disease_name = request.form['DiseaseName']
        livestockClassification = request.form['LivestockClassification']

        cattleDir = base_directory + '/Cattle'
        swineDir = base_directory + '/Swine'
        goatDir = base_directory + '/Goat'

        if not disease_name:
            return jsonify({"success": False, "message": "Invalid disease name"})

        # Get the selected TrainPercentage and ValidationPercentage
        train_percentage = int(request.form['TrainPercentage'])

        # Get the uploaded files
        uploaded_files = request.files.getlist('image')

        if(livestockClassification == "Swine"):
            train_dir = os.path.join(swineDir, 'train', disease_name)
            val_dir = os.path.join(swineDir, 'val', disease_name)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            total_images = len(uploaded_files)
            num_train_images = int(train_percentage / 100 * total_images)
            num_val_images = total_images - num_train_images

            # Check if the disease folder already exists and get a list of existing files
            existing_files = set(os.listdir(train_dir) + os.listdir(val_dir))

            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file:
                    filename = secure_filename(uploaded_file.filename)
                    file_extension = filename.rsplit('.', 1)[1].lower()

                    if file_extension not in allowed_image_extensions:
                        continue  # Skip files with disallowed extensions

                    if i < num_train_images:
                        destination_dir = train_dir
                    else:
                        destination_dir = val_dir

                    # Check for filename conflicts and add a suffix if necessary
                    base, ext = os.path.splitext(filename)
                    new_filename = filename
                    suffix = 1
                    while new_filename in existing_files:
                        new_filename = f"{base}_{suffix}{ext}"
                        suffix += 1

                    existing_files.add(new_filename)  # Update the list of existing files
                    destination_path = os.path.join(destination_dir, new_filename)
                    destination_path = os.path.abspath(destination_path)

                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    # uploaded_file.save(destination_path)

                    # Convert and save the image as JPEG
                    img = Image.open(uploaded_file)
                    img = img.convert("RGB")  # Ensure the image is in RGB mode
                    img.save(destination_path, "JPEG")

            return jsonify({"success": True, "message": "Data received."})
        elif(livestockClassification == "Goat"):
            train_dir = os.path.join(goatDir, 'train', disease_name)
            val_dir = os.path.join(goatDir, 'val', disease_name)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            total_images = len(uploaded_files)
            num_train_images = int(train_percentage / 100 * total_images)
            num_val_images = total_images - num_train_images

            # Check if the disease folder already exists and get a list of existing files
            existing_files = set(os.listdir(train_dir) + os.listdir(val_dir))

            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file:
                    filename = secure_filename(uploaded_file.filename)
                    file_extension = filename.rsplit('.', 1)[1].lower()

                    if file_extension not in allowed_image_extensions:
                        continue  # Skip files with disallowed extensions

                    if i < num_train_images:
                        destination_dir = train_dir
                    else:
                        destination_dir = val_dir

                    # Check for filename conflicts and add a suffix if necessary
                    base, ext = os.path.splitext(filename)
                    new_filename = filename
                    suffix = 1
                    while new_filename in existing_files:
                        new_filename = f"{base}_{suffix}{ext}"
                        suffix += 1

                    existing_files.add(new_filename)  # Update the list of existing files
                    destination_path = os.path.join(destination_dir, new_filename)
                    destination_path = os.path.abspath(destination_path)

                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    # uploaded_file.save(destination_path)

                    # Convert and save the image as JPEG
                    img = Image.open(uploaded_file)
                    img = img.convert("RGB")  # Ensure the image is in RGB mode
                    img.save(destination_path, "JPEG")

            return jsonify({"success": True, "message": "Data received."})
        elif(livestockClassification == "Cattle"):
            train_dir = os.path.join(cattleDir, 'train', disease_name)
            val_dir = os.path.join(cattleDir, 'val', disease_name)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            total_images = len(uploaded_files)
            num_train_images = int(train_percentage / 100 * total_images)
            num_val_images = total_images - num_train_images

            # Check if the disease folder already exists and get a list of existing files
            existing_files = set(os.listdir(train_dir) + os.listdir(val_dir))

            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file:
                    filename = secure_filename(uploaded_file.filename)
                    file_extension = filename.rsplit('.', 1)[1].lower()

                    if file_extension not in allowed_image_extensions:
                        continue  # Skip files with disallowed extensions

                    if i < num_train_images:
                        destination_dir = train_dir
                    else:
                        destination_dir = val_dir

                    # Check for filename conflicts and add a suffix if necessary
                    base, ext = os.path.splitext(filename)
                    new_filename = filename
                    suffix = 1
                    while new_filename in existing_files:
                        new_filename = f"{base}_{suffix}{ext}"
                        suffix += 1

                    existing_files.add(new_filename)  # Update the list of existing files
                    destination_path = os.path.join(destination_dir, new_filename)
                    destination_path = os.path.abspath(destination_path)

                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    # uploaded_file.save(destination_path)

                    # Convert and save the image as JPEG
                    img = Image.open(uploaded_file)
                    img = img.convert("RGB")  # Ensure the image is in RGB mode
                    img.save(destination_path, "JPEG")

            return jsonify({"success": True, "message": "Data received."})




@app.route("/uploadNewDatasets", methods=["POST"])
def uploadNewDatasets():
    if request.method == 'POST':

        disease_name = request.form['DiseaseName']
        livestockClassification = request.form['livestockClassification']

        files = request.files.getlist('image')

        cattleDir = base_directory + '/Cattle'
        swineDir = base_directory + '/Swine'
        goatDir = base_directory + '/Goat'

        if(livestockClassification == "Cattle"):
            train_dir = os.path.join(cattleDir, 'train', disease_name)
            os.makedirs(train_dir, exist_ok=True)

            existing_files = set(os.listdir(train_dir))

            for file in files:
                if file:
                    filename = secure_filename(file.filename)

                    # Check for filename conflicts and add a suffix if necessary
                    base, ext = os.path.splitext(filename)
                    new_filename = filename
                    suffix = 1
                    while new_filename in existing_files:
                        new_filename = f"{base}_{suffix}{ext}"
                        suffix += 1

                    existing_files.add(new_filename)  # Update the list of existing files
                    destination_path = os.path.join(train_dir, new_filename)
                    destination_path = os.path.abspath(destination_path)

                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    file.save(destination_path)

            return jsonify({"success": True, "message": "Data received."})
        elif (livestockClassification == "Goat"):
            train_dir = os.path.join(goatDir, 'train', disease_name)
            os.makedirs(train_dir, exist_ok=True)

            existing_files = set(os.listdir(train_dir))

            for file in files:
                if file:
                    filename = secure_filename(file.filename)

                    # Check for filename conflicts and add a suffix if necessary
                    base, ext = os.path.splitext(filename)
                    new_filename = filename
                    suffix = 1
                    while new_filename in existing_files:
                        new_filename = f"{base}_{suffix}{ext}"
                        suffix += 1

                    existing_files.add(new_filename)  # Update the list of existing files
                    destination_path = os.path.join(train_dir, new_filename)
                    destination_path = os.path.abspath(destination_path)

                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    file.save(destination_path)

            return jsonify({"success": True, "message": "Data received."})
        elif (livestockClassification == "Swine"):
            train_dir = os.path.join(swineDir, 'train', disease_name)
            os.makedirs(train_dir, exist_ok=True)

            existing_files = set(os.listdir(train_dir))

            for file in files:
                if file:
                    filename = secure_filename(file.filename)

                    # Check for filename conflicts and add a suffix if necessary
                    base, ext = os.path.splitext(filename)
                    new_filename = filename
                    suffix = 1
                    while new_filename in existing_files:
                        new_filename = f"{base}_{suffix}{ext}"
                        suffix += 1

                    existing_files.add(new_filename)  # Update the list of existing files
                    destination_path = os.path.join(train_dir, new_filename)
                    destination_path = os.path.abspath(destination_path)

                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    file.save(destination_path)

            return jsonify({"success": True, "message": "Data received."})

        if not disease_name:
            return jsonify({"success": False, "message": "Invalid disease name"})

@app.route("/train", methods=["POST"])
def train_model():

    data = request.get_json()  # Parse JSON data from the request body
    selected_folders = data.get('selectedFolders', [])
    epoch_number = data.get('epoch')
    classification = data.get('livestockClassification')

    if epoch_number is not None:
        # Convert epoch_number to an integer if needed
        try:
            epochs = int(epoch_number)
        except ValueError:
            return jsonify({"error": "Invalid epoch value. Please enter a valid number."})
    else:
        return jsonify({"error": "Epoch value not provided."})

    highest_accuracy = 0.0

    img_height, img_width = 150, 150
    batch_size = 8


    if(classification == "Cattle"):
        train_data_dir = './dataset/Cattle/train'
        val_data_dir = './dataset/Cattle/val'
        checkpoint_filepath = './model/Cattle/best_model.h5'

        model_dir = './model/Cattle'
        os.makedirs(model_dir, exist_ok=True)

        archive_dir = './Archive_models/Cattle'
        os.makedirs(archive_dir, exist_ok=True)

        txt_archive_dir = './Archive_classes/Cattle'
        os.makedirs(txt_archive_dir, exist_ok=True)

        image_dir = './Images/Cattle'
        os.makedirs(image_dir, exist_ok=True)

    elif(classification == "Goat"):
        train_data_dir = './dataset/Goat/train'
        val_data_dir = './dataset/Goat/val'
        checkpoint_filepath = './model/Goat/best_model.h5'

        model_dir = './model/Goat'
        os.makedirs(model_dir, exist_ok=True)

        archive_dir = './Archive_models/Goat'
        os.makedirs(archive_dir, exist_ok=True)

        txt_archive_dir = './Archive_classes/Goat'
        os.makedirs(txt_archive_dir, exist_ok=True)

        image_dir = './Images/Goat'
        os.makedirs(image_dir, exist_ok=True)


    elif (classification == "Swine"):
        train_data_dir = './dataset/Swine/train'
        val_data_dir = './dataset/Swine/val'
        checkpoint_filepath = './model/Swine/best_model.h5'

        model_dir = './model/Swine'
        os.makedirs(model_dir, exist_ok=True)

        archive_dir = './Archive_models/Swine'
        os.makedirs(archive_dir, exist_ok=True)

        txt_archive_dir = './Archive_classes/Swine'
        os.makedirs(txt_archive_dir, exist_ok=True)

        image_dir = './Images/Swine'
        os.makedirs(image_dir, exist_ok=True)



    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        class_names = selected_folders
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        label_mode='categorical',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        class_names=selected_folders
    )

    class_names = train_ds.class_names
    print(class_names)
    print(type(class_names))

    resnet_model = Sequential()

    pretrained_model = ResNet152(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        classes=len(class_names),
        pooling='avg'
    )

    for layer in pretrained_model.layers:
        if isinstance(layer, Conv2D):
            print(f'{layer.name} - filters: {layer.filters}')
        layer.trainable = False

    resnet_model.add(pretrained_model)

    resnet_model.add(Flatten())

    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(256, activation='relu'))
    resnet_model.add(Dense(128, activation='relu'))
    resnet_model.add(Dense(len(class_names), activation='softmax'))

    resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    resnet_model.summary()



    # Check if an old model exists in the model directory and move it to the archive directory
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    new_model_filename = formatted_date + '_best_model.h5'

    old_model_path = os.path.join(model_dir, 'best_model.h5')  # Use the same model name

    if os.path.isfile(old_model_path):
        # Move the old model to the archive directory
        shutil.move(old_model_path, os.path.join(archive_dir, new_model_filename))

    # Txt file Archiving

    # Set up the archive directory path
    # txt_archive_dir = './Archive_classes'
    # os.makedirs(txt_archive_dir, exist_ok=True)

    # Check if an old model exists in the model directory and move it to the archive directory
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    new_txt_filename = formatted_date + '_class_names.txt'

    old_model_path = os.path.join(model_dir, 'class_names.txt')  # Use the same model name

    if os.path.isfile(old_model_path):
        # Move the old model to the archive directory
        shutil.move(old_model_path, os.path.join(archive_dir, new_txt_filename))

    # Create a new class_names.txt file and populate it with class names
    new_class_names_path = os.path.join(model_dir, 'class_names.txt')

    with open(new_class_names_path, 'w') as file:
        for class_name in class_names:
            file.write(f"{class_name}\n")



    # Create a new ModelCheckpoint callback to save the new model
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,  # Save only the best model
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # Save the model when the metric is maximized
        verbose=1  # Display progress
    )

    history = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_checkpoint_callback]
    )

    # Find the highest accuracy from the training history
    highest_accuracy = max(history.history['val_accuracy'])


    new_accuracy_graph_name = formatted_date + "_Current_training_accuracy.png"
    new_loss_graph_name = formatted_date + "Current_training_loss.png"

    # Define the directory and filename for the accuracy graph
    # image_dir = './Images'
    accuracy_graph_filename = 'Current_training_accuracy.png'
    loss_graph_filename = 'Current_training_loss.png'
    training_accuracy_path = os.path.join(image_dir, accuracy_graph_filename)
    training_loss_path = os.path.join(image_dir, loss_graph_filename)

    # Check if the old image file exists
    if os.path.isfile(training_accuracy_path):

        new_training_accuracy_path = os.path.join(image_dir, new_accuracy_graph_name)

        # Rename the old image file
        os.rename(training_accuracy_path, new_training_accuracy_path)

        # Update the path to the new image
        training_accuracy_path = new_training_accuracy_path

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(training_accuracy_path)  # Save the graph to your image directory
    plt.show()  # Display the graph if needed

    # Clear the previous plot to create a new one for loss
    plt.clf()

    # Check if the old image file exists
    if os.path.isfile(training_loss_path):
        new_training_loss_path = os.path.join(image_dir, new_loss_graph_name)

        # Rename the old image file
        os.rename(training_loss_path, new_training_loss_path)

        # Update the path to the new image
        training_loss_path = new_training_loss_path

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(training_loss_path)  # Save the graph to your image directory
    plt.show()  # Display the graph if needed


    data = {
        'accuracy_Rate': highest_accuracy
    }

    # php_api_url = 'https://thefarmer.online/Flask/Flask_API.php' #From Hostinger
    php_api_url = 'http://localhost/Thesis_web/Flask/Flask_API.php'  # From LocalHost
    response = requests.post(php_api_url, data=data)

    if response.status_code == 200:
        print("Data sent and inserted successfully.")
    else:
        print("Failed to send data to the PHP API.")

    # Encode the graphs as base64 strings
    def encode_image_as_base64(image_path):
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_encoded


    accuracy_graph_base64 = encode_image_as_base64(training_accuracy_path)
    loss_graph_base64 = encode_image_as_base64(training_loss_path)

    response_data = {
        "message": "Training completed successfully",
        "selectedFolders": selected_folders,
        "TrainingAccuracy": highest_accuracy,
        "TrainingLossGraph": loss_graph_base64,
        "TrainingAccuracyGraph": accuracy_graph_base64
    }

    return jsonify(response_data)



@app.route("/fetchFolder", methods=["POST"])
def fetchFolder():
    data = request.get_json()
    classification = data.get('classification')

    SwineDir = './dataset/Swine/train'
    CattleDir = './dataset/Cattle/train'
    GoatDir = './dataset/Goat/train'


    if(classification == "Swine"):
        try:
            folder_names = [folder for folder in os.listdir(SwineDir) if
                            os.path.isdir(os.path.join(SwineDir, folder))]
            return jsonify({"folderDir": folder_names})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif(classification == "Cattle"):
        try:
            folder_names = [folder for folder in os.listdir(CattleDir) if
                            os.path.isdir(os.path.join(CattleDir, folder))]
            return jsonify({"folderDir": folder_names})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif (classification == "Goat"):
        try:
            folder_names = [folder for folder in os.listdir(GoatDir) if
                            os.path.isdir(os.path.join(GoatDir, folder))]
            return jsonify({"folderDir": folder_names})
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == "__main__":
    base_directory = "./dataset"
    model_directory = "./model"
    image_dir = "./images"
    # app.run(debug=True)

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

        # Create subdirectories for Swine, Cattle, and Goat within the model directory
        for livestock in ["Swine", "Cattle", "Goat"]:
            livestock_directory = os.path.join(model_directory, livestock)
            os.makedirs(livestock_directory, exist_ok=True)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

        # Create subdirectories for Swine, Cattle, and Goat within the model directory
        for livestock in ["Swine", "Cattle", "Goat"]:
            livestock_directory = os.path.join(image_dir, livestock)
            os.makedirs(livestock_directory, exist_ok=True)

    app.run(host='0.0.0.0', port=5000)

