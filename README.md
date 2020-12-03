# FashionMNIST-PyTorch
> Training my first Neural Network model using FashionMNIST and PyTorch

---

## Description
<p align=justify>
<a href="https://github.com/zalandoresearch/fashion-mnist"> <b>Fashion-MNIST</b></a> is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original <a href="http://yann.lecun.com/exdb/mnist/"> <b>MNIST</b></a> dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

A sample of the dataset look like this: <br>
  <p align="center">
    <img src="https://github.com/mpalaourg/FashionMNIST-PyTorch/blob/main/images/fashion-mnist.png" alt="dataset" width="500" height="500">
  </p>
</p>

## Labels
| Index | Label |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Network Summary
| Layer | Parameter Name | Parameter Value | Parameter Description |
|-------|----------------|-----------------|-----------------------|
| conv1 | in_channels    | 1               | Color channels in the input image |
| conv1 | kernel_size    | 5               | Hyperparameter        |
| conv1 | out_channels   | 6               | Hyperparameter        |
| conv2 | in_channels    | 6               | Number of out_channels in prev layer |
| conv2 | kernel_size    | 5               | Hyperparameter        |
| conv2 | out_channels   | 12              | Hyperparameter (higher than prev conv layer) |
| fc1 | in_features      | 12x4x4          | Flattened output from prev layer |
| fc1 | out_features     | 120             | Hyperparameter        |
| fc2 | in_features      | 120             | Number of out_features of prev layer |
| fc2 | out_features     | 60              | Hyperparameter (lower than prev linear layer) |
| out | in_features      | 60              | Number of out_channels in prev layer | 
| out | out_features     | 10              | Number of prediction classes |

---

## Results
Two hyperparameters, `batch_size` and `lr`, will be explored. Initially, for 10 `epochs` and values:
```
  batch_size = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
  lr         = [.01, .001, .0001, .00001]
```
better results were achieved based on accuracy (`89.18%`) were for `batch_size=10, lr= 0.001`. <br>
For the final results, the technique of Standardization was explored, to see if the results will be better. For 20 `epochs` and values:
```
  batch_size = [10, 20, 50, 100]
  lr         = [.01, .001]
  trainset   = ['not_normal', 'normal']
```
where the results were better for `batch_size=20, lr= 0.001, trainset='normal'`, with `93.71%` accuracy. <br>
Finally, for the best parameters the network was trained and the confusion matrix of the `test_set` is given below: <br>
  <p align="center">
    <img src="https://github.com/mpalaourg/FashionMNIST-PyTorch/blob/main/images/confusion_matrix.png" alt="cm" width="500" height="500">
  </p>

Accuracy on `train data`: `94.42%` <br>
Accuracy on `test data` : `89.23%`

---

## Dependecies 
```
torch       (version: 1.7.0)
torchvision (version: 0.8.1)

pip install matplotlib
pip install numpy
pip install scikit-learn
pip install pandas
```

---

## Support 

Reach out to me:

- [mpalaourg's email](mailto:gbalaouras@gmail.com "gbalaouras@gmail.com")

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpalaourg/FashionMNIST-PyTorch/blob/main/LICENSE)
