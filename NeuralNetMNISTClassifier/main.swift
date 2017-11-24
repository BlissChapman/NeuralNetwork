//
//  main.swift
//  NeuralNetMNISTClassifier
//
//  Created by WILLIAM CHAPMAN on 11/21/17.
//  Copyright © 2017 WILLIAM CHAPMAN. All rights reserved.
//

import Foundation

let fileImagePath = "/Users/chapman/Desktop/Machine Learning/MNIST.database.digit.database/train-images-idx3-ubyte"
let fileLabelPath = "/Users/chapman/Desktop/Machine Learning/MNIST.database.digit.database/train-labels-idx1-ubyte"

let imageByteData: [UInt8]
let labelByteData: [UInt8]

do {
    imageByteData = try MNISTUtils.readByteData(fromPath: fileImagePath)
    labelByteData = try MNISTUtils.readByteData(fromPath: fileLabelPath)
} catch {
    if let error = error as? MNISTUtils.MNISTUtilsError {
        switch error {
        case .invalidPath(let errorDescription):
            print(errorDescription)
        }
    }
    exit(1)
}

guard let headerInfo = MNISTUtils.extractImageHeaderInfo(fromByteData: imageByteData) else {
    print("Could not extract images header info from byte data.")
    exit(2)
}

guard let images = MNISTUtils.readImages(fromByteData: imageByteData, withHeaderInfo: headerInfo) else {
    print("Cannot extract images from byte data")
    exit(3)
}

guard let labelHeaderInfo = MNISTUtils.extractLabelHeaderInfo(fromByteData: labelByteData) else {
    print("Could not extract label header info from byte data.")
    exit(4)
}

guard let labels = MNISTUtils.readLabels(fromByteData: labelByteData, withHeaderInfo: labelHeaderInfo) else {
    print("Cannot extract labels from byte data")
    exit(5)
}

print(headerInfo)
for (index, image) in images.enumerated() {
    print("-----------------------------------------------------------------")
    print("                       \(labels[index])")
    print("-----------------------------------------------------------------")
    MNISTUtils.printImageGrid(image: image)
}

