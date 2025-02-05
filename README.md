# Number Classifier

This program is designed to create and train neural networks using Keras, allowing users to configure custom models to classify handwritten digits from the MNIST dataset, with prediction capabilities for user-drawn numbers.
The backend handles model training, loading, and inference, while a simple Tkinter interface provides an accessible way to interact with the system.

![imagen](https://github.com/user-attachments/assets/0735d2f1-e4a0-4983-8313-aadc0f2446fc)


## Features
- **Train a Model**: Configure custom layers and train a neural network on the MNIST dataset.
- **Load a Model**: Load pre-trained models in `.h5` format.
- **Real-time Prediction**: Draw a number on a canvas and let the model classify it.
- **Save Trained Models**: Save trained models for later use.

## Installation
### Prerequisites
Ensure you have Python installed (version 3.7 or higher).

### Install Dependencies
Run the following command to install the required packages:
```sh
pip install -r requirements.txt
```

## Usage
### Running the Program
Execute the following command:
```sh
python main.py
```

### Screens & Functionality
#### Main Menu
- **Train Model**: Opens a screen to configure and train a new model.
- **Load Model**: Allows you to load a pre-trained `.h5` model.
- **Exit**: Closes the application.

#### Training Screen
- Add and configure layers (Dense, Conv2D, MaxPooling2D).
- Set activation functions and parameters.
- Define the number of training epochs.
- Start the training process with real-time progress updates.

![imagen](https://github.com/user-attachments/assets/7a119880-4c44-4572-bf77-5599b6adeeac)


#### Prediction Screen
- Draw a digit on the canvas.
- Click "Predict" to classify the drawn number.
- View the model's confidence scores.
- Save the trained model.

![imagen](https://github.com/user-attachments/assets/08ceff4d-19e6-4b5d-b254-c37b6d1d3874)

