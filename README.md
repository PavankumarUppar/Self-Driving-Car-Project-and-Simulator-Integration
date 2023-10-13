# Self-Driving Car Project and Simulator Integration

This repository contains code for a self-driving car project, including the integration of the model with the Udacity Self-Driving Car Simulator. The project involves collecting and preprocessing data, training a neural network model, and using it to control a car autonomously within the simulator environment.

## Data Collection and Preparation

The project relies on a dataset that includes images from the car's perspective and corresponding steering angles. The data is organized in a CSV file with columns such as 'center,' 'left,' 'right,' 'steering,' 'throttle,' 'reverse,' and 'speed.' The dataset is read and preprocessed using Python and several libraries such as Pandas, OpenCV, and Matplotlib.

## Data Augmentation

Data augmentation is essential to increase the variability of the dataset. The `random_augment` function applies random transformations to the images, including panning, zooming, brightness adjustment, and horizontal flipping.

## Data Preprocessing

The `img_preprocess` function processes the images to make them suitable for input into the neural network. This involves cropping, converting to the YUV color space, applying Gaussian blur, resizing, and normalizing the pixel values.

## Model Architecture

The main neural network model is based on the NVIDIA architecture for self-driving cars. It consists of several convolutional layers followed by fully connected layers. The architecture is defined using the Keras library. The model is designed to predict the steering angle from the input images.

## Training the Model

The `batch_generator` function generates batches of training data for the model. It randomly selects images from the dataset, applies augmentations, and preprocesses them. The model is trained using the generated batches with the specified number of training epochs. Training and validation data generators are created to feed data to the model during training.

The model is compiled with the mean squared error (MSE) loss function and the Adam optimizer. The training process is then initiated using the `model.fit` method. Training progress, including loss values, is logged.

## Self-Driving Car Simulator Integration

In addition to training the model, this repository provides code to integrate your model with the Udacity Self-Driving Car Simulator. The Udacity Self-Driving Car Simulator is a powerful tool for testing and validating autonomous driving algorithms.

## Prerequisites

Before running the integration code, make sure to have the following:

1. **Udacity Self-Driving Car Simulator:** Download and install the simulator from the Udacity GitHub repository: [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim).
2. **Trained Model:** You should have a pre-trained model in H5 format, capable of predicting steering angles based on input images. Update the code with the correct path to your model.

## Usage

1. To use this code to integrate your self-driving car model with the Udacity Self-Driving Car Simulator:
2. Ensure you have the Udacity Self-Driving Car Simulator installed.
3. Train your model or obtain a pre-trained model in H5 format.
4. Update the load_model function with the correct path to your model file.
5. Run the code. It will create a server that listens for connections from the simulator.
6. Start the Udacity Self-Driving Car Simulator, select a track, and choose "Autonomous Mode."
7. Your model will now control the car within the simulator, sending control commands based on the input images and telemetry data.

Observe how well your model performs in the simulator and make adjustments to your model or control logic as needed.

Remember that this code assumes the Udacity Self-Driving Car Simulator is running in autonomous mode, and it will send control commands to the simulator to drive the car based on your model's predictions.

Enjoy testing and fine-tuning your self-driving car model with the simulator! If you have any questions or need further assistance, feel free to reach out.




