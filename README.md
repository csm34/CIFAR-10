# CIFAR-10

## Project Overview
This project demonstrates a comprehensive comparison of various machine learning and deep learning models applied to the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training and 10,000 test images. This project evaluates the performance of both classical machine learning algorithms and a Convolutional Neural Network (CNN) for image classification.

## Dataset
- **Dataset**: CIFAR-10
- **Training Images**: 50,000
- **Test Images**: 10,000
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Algorithms Implemented
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Random Forest**
4. **Naive Bayes**
5. **K-Means Clustering**
6. **Convolutional Neural Network (CNN)**

## Preprocessing Steps
1. **Flattening Images**: For machine learning algorithms, the 32x32x3 images are flattened into a 1D vector.
2. **Normalization**: Pixel values are normalized to the range [0, 1].
3. **Standardization**: Applied to data for classical machine learning models.
4. **One-Hot Encoding**: Used for labels in CNN.

## Model Implementations
### Classical Machine Learning Models
- **K-Nearest Neighbors**: Classifies based on the majority vote of nearest neighbors.
- **Logistic Regression**: Performs multi-class classification using a softmax function.
- **Random Forest**: An ensemble learning method using decision trees.
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem.
- **K-Means Clustering**: An unsupervised algorithm clustering data into 10 clusters.

### Deep Learning Model
- **Convolutional Neural Network (CNN)**:
  - Layers:
    - 2D Convolutional layers
    - Max Pooling layers
    - Fully connected Dense layers
    - Dropout for regularization
  - Activation Functions:
    - ReLU for intermediate layers
    - Softmax for output layer
  - Optimizer: Adam
  - Loss Function: Categorical Crossentropy


### Performance Visualization
- Bar graphs comparing accuracy of models.
- Training and validation accuracy graph for CNN.

## How to Run the Code
1. Install required libraries:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy
   ```
2. Run the Python script.
3. The results will be displayed as:
   - Model accuracy printed in the console.
   - Bar graphs for performance comparison.
   - CNN training and validation accuracy plot.

## Output
- Model accuracy scores for each algorithm.
- Performance comparison graphs.
- Saved CNN model as `cifar10_cnn_model.keras`.


## License
This project is open-source and available under the MIT License.

## Acknowledgements
- CIFAR-10 dataset by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
- TensorFlow and scikit-learn libraries.
