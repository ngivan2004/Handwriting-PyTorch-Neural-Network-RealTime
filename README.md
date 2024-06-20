# Real time Interactive MNIST Handwriting Classification Neural Network trained with PyTorch
## Live demo: https://aihandwriting.netlify.app/

### Web Version Inference (Using Onnxjs):
https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/81a6377b-5074-479b-aed9-87fd3174efc9

### Python Version (MatPlotLib):
https://github.com/ngivan2004/Handwriting-PyTorch-Neural-Network-RealTime/assets/61515871/a0eb1096-b779-4001-a433-1820c16621af

#### I recently discovered Onnxjs, which allows deploying PyTorch models using JavaScript. This inspired me to create a web demo version of my project that runs locally through JavaScript. While I utilized generative AI extensively for the website design, I must credit [Elliot Waite](https://www.youtube.com/@elliotwaite), whose code (particularly for the painting canvas) heavily influenced my work. However, the models used throughout the project are primarily designed, implemented, and trained by me.

This is a small personal project on exploring PyTorch, creating and training my own neural network. It is a multilayer perception network (MLP) with 2 hidden layers, and ReLU for activation functions. A softMax function is used to generate a probability distribution from 1 through 9 based on the handwriting of the user and it updates in real-time as the user is writing. To begin, install the required libraries on ```requirements.txt``` and also make sure ```tk``` is working. Run ```model_train.py``` for training, and try the augmented versions too as I have added some transformations to them to make the data more difficult to train on, thereby theoretically producint better outputs. For inference, run ```model_inference```.
