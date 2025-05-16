# DEEP-LEARNING-PROJECT
Company Name:CODTECH IT SOLUTIONS

Name: Ayushi Verma

Intern ID :CT6MTNYD

Domain: Data Science 

Durations: 6 months

Mentor:Muzammil Ahmed

# Description of the Task 2 Deep Learning Project:
Task Description: Deep Learning Pipeline for Image Classification using CNN
This task involves building a deep learning project pipeline to classify images using a Convolutional Neural Network (CNN). The primary goal is to develop a robust image classification model using TensorFlow and Keras that can accurately categorize images into their respective classes. The project follows a structured approach: loading and preprocessing image data, building a CNN architecture, training and evaluating the model, and visualizing performance metrics.

1. Dataset Preparation
The image dataset is organized in a directory format where each subfolder represents a class label. The data is loaded using TensorFlow’s image_dataset_from_directory utility, which automates labeling and batching. The dataset is split into training, validation, and testing subsets to ensure generalization of the model.

2. Data Preprocessing
To standardize the input for the CNN, images are resized to a consistent shape (e.g., 180x180 pixels) and normalized. The data pipeline includes:
Rescaling pixel values to a [0, 1] range using Rescaling layer.
Optional data augmentation using layers such as RandomFlip and RandomRotation to improve model robustness and reduce overfitting.

3. Model Building
A Convolutional Neural Network (CNN) is defined using the Keras Sequential API. The architecture typically includes:
Convolutional layers with ReLU activation to extract image features.
MaxPooling layers to reduce spatial dimensions.
Dropout layers to prevent overfitting.
Dense layers to learn complex patterns.
A final softmax output layer to classify images into multiple categories.

4. Model Compilation and Training
The model is compiled with the Adam optimizer and categorical crossentropy loss function, suitable for multi-class classification tasks. It is trained over several epochs, and accuracy and loss are tracked for both training and validation sets. EarlyStopping or ModelCheckpoint callbacks may be used to optimize training.

5. Model Evaluation
After training, the model is evaluated on a separate test set to measure its accuracy and generalization ability. A classification report is generated, providing metrics like precision, recall, and F1-score for each class. A confusion matrix is plotted to visualize prediction performance across all classes.

6. Performance Visualization
Graphs are plotted for:
Training and validation accuracy across epochs.
Training and validation loss across epochs.
These plots help in diagnosing underfitting or overfitting and understanding the model’s learning behavior.

7. Prediction and Testing
The model is used to predict classes of new or unseen images. This step involves loading a test image, preprocessing it to match the model’s input shape, and feeding it into the trained model for prediction. The predicted class label is then displayed.

![image](https://github.com/user-attachments/assets/6596b262-6c12-4c2b-9541-788ef7eb5861)
