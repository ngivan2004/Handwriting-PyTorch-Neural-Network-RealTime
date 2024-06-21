# Real time Interactive MNIST Handwriting Classification Neural Network trained with PyTorch
## Live demo: https://aihandwriting.netlify.app/

### Web Version Inference (Using Onnxjs):
https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/d08a164f-fdb5-4ea4-bea9-5e7359fbd745
### Python Version (MatPlotLib):
https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/a0eb1096-b779-4001-a433-1820c16621af


#### Update 2: I have now implemented both an MLP and a CNN model, and I have also implemented augmented data in training with rotations, sheers, transformations to increase the complexity of training data. Both models are now live on the website.

#### Update 1: I recently discovered Onnxjs, which allows deploying PyTorch models using JavaScript. This inspired me to create a web demo version of my project that runs locally through JavaScript. While I utilized generative AI extensively for the website design, I must credit [Elliot Waite](https://www.youtube.com/@elliotwaite), whose code (particularly for the painting canvas) heavily influenced my work. However, the models used throughout the project are primarily designed, implemented, and trained by me.

This is a personal project on exploring PyTorch, creating and training my own neural networks. There are 2 networks trained.


1. A multilayer perception network (MLP) with 2 hidden layers, and ReLU for activation functions trained on standard MNIST.
2. A Convolutional neural network (CNN) trained on augmented MNIST with random rotations, translations, sheers.


A softMax function is used to generate a probability distribution from 1 through 9 based on the handwriting of the user and it updates in real-time as the user is writing. To begin, install the required libraries on ```requirements.txt``` (currently missing)  and also make sure ```tk``` is working. Run ```model_train.py``` for training, and try the augmented versions too as I have added some transformations to them to make the data more difficult to train on, thereby theoretically producint better outputs. The augmented versions also run on a CNN network instead of an MLP. For inference, run ```model_inference```.

| Model | Test Data | Inputs Tested | Accuracy |
|-------|-----------|---------------|----------|
| MLP   | Standard MNIST | 10,000        | 97.98%   |
| MLP   | Augmented MNIST | 10,000        | 17.49%   |
| CNN   | Standard MNIST | 10,000        | 97.64%   |
| CNN   | Augmented MNIST | 10,000        | 91.27%   |

