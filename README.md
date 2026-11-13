# Developing_Multi-Input_Models_For_OCR
Developed a multi-input model to classify characters from scanned documents

## Multi-Input Deep Learning for Insurance ID Classification: PyTorch CNN on Image + Document Type | Python, PyTorch

Built a multi-input PyTorch model that takes a 64x64 scanned ID-code image and the insurance type (home, life, auto, health, other) and predicts whether the code is a primary or secondary ID, to help digitise insurance claim documents.

# CONTEXT
An insurer digitising historical paper claims needed scanned IDs labelled as primary or secondary. Combining the image with document metadata gives the model context the image alone lacks.

# MY ROLE
I designed the architecture and wrote the training code. The dataset and its loading utilities were provided.

# WHAT I DID
• Designed a two-branch network: a CNN branch (Conv2d 3x3, pooling, flatten, linear) for the image and a fully connected branch for the one-hot insurance type
• Concatenated both feature vectors into a classifier head for the two ID classes
• Used Adam (lr 0.001) with cross-entropy loss and mini-batches of 10 through a standard PyTorch training loop
• Trained for 20 epochs and monitored loss per epoch (final loss: 0.01)

# RESULTS
• Working multi-modal model with image and type inputs
• Training loss fell from 0.98 to 0.01 across 20 epochs

# TECH STACK
Python, PyTorch, torchvision-style CNN layers, NumPy, Matplotlib
