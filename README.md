Vision Transformer (ViT) Animal Image Classifier

Project Overview
This project presents an advanced image classification solution leveraging the Vision Transformer (ViT) architecture. Our goal is to accurately identify and categorize various animal species from a specific dataset, showcasing the powerful capabilities of transformers in computer vision tasks. Unlike traditional convolutional neural networks, the ViT processes images by treating them as sequences of visual "words" (patches), enabling it to capture global contextual information and long-range dependencies, which is crucial for distinguishing between diverse animal classes.

Dataset
This classifier is trained and evaluated on a custom animal kingdom dataset. This comprehensive collection comprises approximately 3400 images distributed across 13 distinct animal categories, including lions, tigers, etc... 

Key Features
Vision Transformer (ViT) Architecture: Implements a state-of-the-art transformer model adapted for image classification.

High Accuracy: Achieves robust performance in accurately classifying various animal species within the dataset.

Scalable Design: The modular nature of the ViT allows for potential scaling to larger and more complex datasets.

Interpretability (Optional/If applicable): The attention mechanisms within the ViT offer insights into which parts of the image the model focuses on for classification.

Technical Details
The core of this project is the Vision Transformer (ViT). Here's a brief breakdown of its methodology:

Patch Embedding: Input images are first divided into fixed-size non-overlapping patches. Each patch is then flattened and linearly projected into a higher-dimensional embedding space.

Positional Encoding: To retain spatial information lost during patch flattening, learnable positional embeddings are added to the patch embeddings.

Transformer Encoder: The sequence of patch embeddings, along with a learnable [class] token, is fed into a standard Transformer encoder. This encoder consists of multiple layers, each containing:

Multi-Head Self-Attention: Allows the model to weigh the importance of different patches relative to each other, capturing global relationships.

Multi-Layer Perceptron (MLP): A feed-forward network applied independently to each position.

Classification Head: The final hidden state of the [class] token from the Transformer encoder is fed into a simple MLP head for final classification.
