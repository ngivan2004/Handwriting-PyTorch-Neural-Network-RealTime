# Real-time Interactive MNIST Handwriting Classification Neural Network trained with PyTorch

## Live demo: [https://aihandwriting.netlify.app/](https://aihandwriting.netlify.app/)

### Web Version Inference (Using Onnxjs):
https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/36b67330-74c9-4eb6-8ee1-b1d81c5d4795
### Python Version (MatPlotLib):
https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/b1c2a5a9-ba62-4a2c-ba40-ddcc753e53be
## Project Updates

### Update 2

I have now implemented both an MLP and a CNN model, and I have also used augmented data in training with rotations, shears, and transformations to increase the complexity of the training data. Both models are now live on the website.

### Update 1

I recently discovered Onnxjs, which allows deploying PyTorch models using JavaScript. This inspired me to create a web demo version of my project that runs locally through JavaScript. While I must admit that I utilized generative AI extensively for the website development, I must credit [Elliot Waite](https://www.youtube.com/@elliotwaite), whose code (particularly for the painting canvas) heavily influenced my work. Note that the models used throughout the project are primarily designed, implemented, and trained by me.

## Project Overview

This is a personal project exploring PyTorch, where I created and trained my own neural networks. There are two networks trained:

1. A Multilayer Perceptron (MLP) network trained on standard MNIST.
2. A Convolutional Neural Network (CNN) trained on augmented MNIST with random rotations, translations, and shears.

### MLP Model

The MLP model is defined as follows:

- **Flatten Layer**: Converts the 28x28 image into a 1D tensor with 784 elements.
- **First Fully Connected Layer**: 784 inputs to 512 outputs, followed by a ReLU activation function.
- **Second Fully Connected Layer**: 512 inputs to 512 outputs, followed by a ReLU activation function.
- **Output Layer**: 512 inputs to 10 outputs (one for each digit).

### CNN Model

The CNN model is defined as follows:

- **First Convolutional Layer**: 1 input channel, 32 output channels, 3x3 kernel size, followed by ReLU activation and max pooling (2x2).
- **Second Convolutional Layer**: 32 input channels, 64 output channels, 3x3 kernel size, followed by ReLU activation and max pooling (2x2).
- **Flatten Layer**: Converts the output of the convolutional layers into a 1D tensor.
- **First Fully Connected Layer**: 64 _ 7 _ 7 inputs to 128 outputs, followed by a ReLU activation function and dropout (0.5).
- **Output Layer**: 128 inputs to 10 outputs (one for each digit).

| Model | Test Data       | Inputs Tested | Accuracy |
| ----- | --------------- | ------------- | -------- |
| MLP   | Standard MNIST  | 10,000        | 97.98%   |
| MLP   | Augmented MNIST | 10,000        | 17.49%   |
| CNN   | Standard MNIST  | 10,000        | 97.64%   |
| CNN   | Augmented MNIST | 10,000        | 91.27%   |

A softmax function is used to generate a probability distribution from 0 through 9 based on the user's handwriting, updating in real-time as the user writes.

### A Note on Data Augmentation

Data augmentation is a technique used to increase the diversity of your training dataset without actually collecting new data. It involves applying random transformations to your existing data, which helps to prevent overfitting and improves the model's generalization capabilities.

In the context of our handwriting recognition project, we apply several transformations to the MNIST dataset, including random rotations, translations, scaling, and shearing. These transformations simulate different writing styles and conditions, making the model more robust when it encounters varied handwriting samples.

Below are two grids for comparison:

- The first grid shows original images from the MNIST dataset.
- The second grid shows the same images after data augmentation has been applied.

<p align="center">
    <img src="https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/bfa8a851-9f57-495f-8d4d-7a8c1f59a3b4" alt="Original Images" width="45%">
    <img src="https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/bc2e4ed2-201f-40d2-ab69-87807d24ba82" alt="Augmented Images" width="45%">
</p>

By comparing these grids, you can see how the augmentations introduce variations while retaining the essential characteristics of the original digits. This enhanced dataset allows the model to learn from a broader range of examples, leading to better performance on unseen data.

## Repository Structure

Here's an overview of the repository structure:

- **website_version/**: Contains the web version of the project.

  - `index.html`: Main HTML file for the web interface.
  - `onnx_model_augmented+cnn.onnx`: ONNX model file for the CNN model trained on augmented data.
  - `onnx_model.onnx`: ONNX model file for the MLP model trained on standard data.
  - `script.js`: JavaScript file for handling the web interface logic.
  - `style.css`: CSS file for styling the web interface.

- **model_conversion_onnx.py**: Script for converting both models to ONNX format. The output will be stored at the same directory of the script.

- **model_inference_augmented+cnn.py**: Script for running inference with the augmented CNN model.

- **model_inference.py**: Script for running inference with the standard model.

- **model_state_augmented+cnn.pth**: Model state file for the augmented CNN.

- **model_state.pth**: Model state file for the standard model.

- **model_train_augmented+cnn.py**: Script for training the augmented CNN model.

- **model_train.py**: Script for training the standard model.

- **model.py**: Contains model definitions.

- **models_eval.py**: Script for evaluating the models.

- **nice_to_haves.py**: Additional scripts or utilities that might be useful.

- **onnxdiagnose.py**: Script for diagnosing issues with ONNX models.

- **README.md**: This readme file.

- **requirements.txt**: The libraries needed for this project.

- **sample_photo_grid.py**: Generates grids of sample training data of MNIST, both standard and augmented with transformations.

## Getting Started

To get started with this project, follow the steps below:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime.git
   cd Handwriting-PyTorch-Neural-Network-RealTime
   ```

2. **Install the required libraries:**
   Make sure you have Python installed, then install the required libraries using `requirements.txt`:

   ```sh
   pip install -r requirements.txt
   ```

   Additionally, ensure that `tkinter` is working properly on your system.

3. **Training the models:**
   Run the following script to train the models. The augmented+cnn versions add transformations and make the data more challenging to train on, and it also trains the data on the CNN model instead of the MLP, producing better outputs:

   ```sh
   python model_train.py
   python model_train_augmented+cnn.py
   ```

4. **Inference:**
   For inference, run the following script:

   ```sh
   python model_inference.py
   python model_inference_augmented+cnn.py
   ```

5. **Running the web demo:**
   To run the web demo locally, you must start a web server. This is necessary because directly opening the `index.html` file in your browser can lead to issues with CORS (Cross-Origin Resource Sharing) policies. You can easily start a local server using Python.

   For Python 3.x, run the following command in the terminal from the project directory:

   ```sh
   cd website_version
   python -m http.server 8000
   ```

   This will start a local server at `http://localhost:8000`. Open your web browser and navigate to `http://localhost:8000/index.html` to view the demo.

   **Note:** If you have trained your own models and converted them into ONNX using the provided training and conversion scripts, and want to use them instead of the pre-trained models, make sure to move the converted ONNX models into the `website_version` folder before running the demo.

## Known Issues

### Drawing Looks Unusual on iPad

There is an observed issue where strokes do not fully render on iPads, resulting in only a thin line being displayed. Interestingly, this problem does not occur on other touch control devices like mobile phones. Notably, the issue disappears and the drawing renders properly when the screen is resized.

- **Impact**: This issue does not affect the actual model performance. The model input remains accurate, as shown in the bottom left-hand corner.
- **Resolution**: The issue resolves itself when the screen is resized. A fix may be implemented in the future if a solution is found and time permits. Touch control was not the primary focus of this project and was added as an additional feature for fun.
