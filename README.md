# ics435-assignment4

## Part1 - Convolutional Neural Networks (CNN) for FashionMNIST
classify images from the FashionMNIST dataset using CNN

### Requirements
pip install torch torchvision scikit-learn matplotlib

### Usage
python fashion_mnist_cnn.py


## Part2 - Fine-Tuning GPT-2 for Joke Generation
Fine-tunes GPT-2 to generate a complete joke from any 3 starting words.

### Requirements
pip install transformers torch accelerate

### Usage
#### Train the model
-     python joke.py --train
Or with more epochs for better quality:
-     python joke.py --train --epochs 10

#### Generate jokes
    python joke.py --generate --prompt "What did the"
    python joke.py --generate --prompt "Why did the"
