A 2017 Thanksgiving break project consisting of a handwritten neural net to classify the MNIST data set.
This project was written in an effort to more intimately understand how neural nets work.

### Running
```
swiftc main.swift MNISTUtils.swift NeuralNetwork.swift
```
```
./main
```

### Results
The best results I had the time/patience to achieve were obtained with a fully-connected
architecture with no hidden layers (i.e. [784, 10]) and a learning rate of 0.001:

```
0: 96.530612244898%
1: 95.7709251101322%
2: 83.1395348837209%
3: 86.7326732673267%
4: 90.020366598778%
5: 71.1883408071749%
6: 91.6492693110647%
7: 87.7431906614786%
8: 83.3675564681725%
9: 80.4757185332012%

OVERALL: 86.9%
```

### MNIST Dataset
Format of the dataset: http://yann.lecun.com/exdb/mnist/

Handwritten Image:
```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
```

Labels (answers):
```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
```

The labels values are 0 to 9.
