

# **\<ins\> Breast Cancer Classification with Neural Network 🩺\</ins\>**

## Overview

This project implements a simple Neural Network model using TensorFlow and Keras to classify breast tumors as Malignant or Benign. It utilizes the Breast Cancer Wisconsin (Diagnostic) dataset from `sklearn.datasets`. The notebook covers data loading, preprocessing, model building, training, evaluation, and prediction on new data.

## Features

  * **Dataset:** Breast Cancer Wisconsin dataset (`sklearn.datasets.load_breast_cancer`)
  * **Model:** Sequential Neural Network with Flatten, Dense (ReLU), and Dense (Sigmoid) layers.
  * **Preprocessing:** Data scaling using `StandardScaler`.
  * **Evaluation:** Accuracy and loss metrics on training, validation, and test sets.
  * **Visualization:** Plots for training and validation accuracy and loss over epochs.
  * **Prediction:** Predicts the class (Malignant or Benign) for new input data.

## Dependencies

To run this project, you need the following Python libraries:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `scikit-learn`
  * `tensorflow`
  * `keras` (integrated within TensorFlow)

Install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Project Structure

```
Breast Cancer Classification with NN.ipynb: Jupyter Notebook containing the entire project code.
```

## How to Run

1.  **Open the Jupyter Notebook:**
    Open the `Breast Cancer Classification with NN.ipynb` file in Google Colab or your local Jupyter environment.

2.  **Run the Notebook:**
    Execute the cells in the notebook sequentially. The notebook will:
    * Load the dataset.
    * Preprocess the data (split into training and testing sets, scale features).
    * Define and compile the Neural Network model.
    * Train the model on the training data with a validation split.
    * Plot the training and validation accuracy and loss.
    * Evaluate the model on the test data.
    * Demonstrate prediction on a sample input.

## Code Breakdown

### Data Loading and Preprocessing

  * Loads the Breast Cancer dataset using `sklearn.datasets.load_breast_cancer`.
  * Creates a pandas DataFrame from the features and target labels.
  * Splits the data into training (80%) and testing (20%) sets using `train_test_split`.
  * Scales the features using `StandardScaler` to standardize the data.

### Model Building

  * Defines a sequential Neural Network model using `keras.Sequential`:
    * `Flatten` layer to convert the 2D input into a 1D array.
    * `Dense` layer with 20 neurons and ReLU activation function.
    * `Dense` output layer with 2 neurons and Sigmoid activation function (for binary classification, although `sparse_categorical_crossentropy` implies integer labels 0 and 1).

### Model Compilation

  * Compiles the model using the `adam` optimizer, `sparse_categorical_crossentropy` loss function (suitable for integer target labels), and `accuracy` as the evaluation metric.

### Model Training

  * Trains the model using the `fit` method on the scaled training data (`X_train_std`) and corresponding labels (`Y_train`).
  * Includes a `validation_split` of 0.1 to monitor the model's performance on unseen data during training.
  * Trains for 10 epochs.

### Model Evaluation

  * Evaluates the trained model on the scaled test data (`X_test_std`) and corresponding labels (`Y_test`) using the `evaluate` method.
  * Prints the test loss and accuracy.

### Prediction

  * Demonstrates how to make predictions on new, unseen data:
    * Defines a sample `input_data`.
    * Converts the input data to a NumPy array and reshapes it to match the model's input shape.
    * Scales the input data using the same `scaler` fitted on the training data.
    * Uses the `predict` method to get the probability distribution over the classes.
    * Uses `np.argmax` to get the predicted class label (0 or 1).
    * Prints the raw prediction probabilities and the final predicted label.
    * Provides a user-friendly output indicating whether the tumor is Malignant or Benign based on the prediction.

## Results

  * The notebook displays plots of training and validation accuracy and loss over the epochs, allowing you to visualize the model's learning progress.
  * The final evaluation on the test set provides the test accuracy and loss, indicating the model's generalization performance.
  * The example prediction demonstrates how to classify new data points using the trained model.

## Future Improvements

  * Experiment with different network architectures (e.g., more layers, different numbers of neurons).
  * Tune hyperparameters such as the learning rate, number of epochs, and batch size.
  * Explore other activation functions and optimizers.
  * Implement techniques to prevent overfitting, such as dropout or regularization.
  * Consider using cross-validation for a more robust evaluation of the model's performance.

## License

This project is likely for educational purposes and doesn't explicitly mention a license. You can consider adding an MIT License for open-source use.

## Contact

For questions or contributions, you can engage through the platform where this notebook is shared (e.g., Google Colab comments).
