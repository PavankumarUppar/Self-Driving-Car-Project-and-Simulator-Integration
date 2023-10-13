# Self-Driving Car Project and Simulator Integration

This repository contains code for a self-driving car project, including the integration of the model with the Udacity Self-Driving Car Simulator. The project involves collecting and preprocessing data, training a neural network model, and using it to control a car autonomously within the simulator environment.

## Data Collection and Preparation

The project relies on a dataset that includes images from the car's perspective and corresponding steering angles. The data is organized in a CSV file with columns such as 'center,' 'left,' 'right,' 'steering,' 'throttle,' 'reverse,' and 'speed.' The dataset is read and preprocessed using Python and several libraries such as Pandas, OpenCV, and Matplotlib.

## Data Augmentation

Data augmentation is essential to increase the variability of the dataset. The `random_augment` function applies random transformations to the images, including panning, zooming, brightness adjustment, and horizontal flipping.

