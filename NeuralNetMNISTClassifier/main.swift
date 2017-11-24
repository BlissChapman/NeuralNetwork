//
//  main.swift
//  NeuralNetMNISTClassifier
//
//  Created by WILLIAM CHAPMAN on 11/21/17.
//  Copyright © 2017 WILLIAM CHAPMAN. All rights reserved.
//

import Foundation

guard let trainingImagesPath = URL(string: "train-images.idx3-ubyte")?.absoluteString,
    let trainingLabelsPath = URL(string: "train-labels.idx1-ubyte")?.absoluteString,
    let testingImagesPath = URL(string: "t10k-images.idx3-ubyte")?.absoluteString,
    let testingLabelsPath = URL(string: "t10k-labels.idx1-ubyte")?.absoluteString else {
        print("Could not retrieve paths to MNIST data.")
        exit(1)
}

let trainingImageByteData: [UInt8]
let trainingLabelByteData: [UInt8]
let testingImageByteData: [UInt8]
let testingLabelByteData: [UInt8]

do {
    trainingImageByteData = try MNISTUtils.readByteData(fromPath: trainingImagesPath)
    trainingLabelByteData = try MNISTUtils.readByteData(fromPath: trainingLabelsPath)
    testingImageByteData = try MNISTUtils.readByteData(fromPath: testingImagesPath)
    testingLabelByteData = try MNISTUtils.readByteData(fromPath: testingLabelsPath)
} catch {
    if let error = error as? MNISTUtils.MNISTUtilsError {
        switch error {
        case .invalidPath(let errorDescription):
            print(errorDescription)
        }
    }
    exit(2)
}

// EXTRACT TRAINING DATA

guard let trainingImageHeaderInfo = MNISTUtils.extractImageHeaderInfo(fromByteData: trainingImageByteData) else {
    print("Could not extract training images header info from byte data.")
    exit(3)
}

guard let trainingImages = MNISTUtils.readImages(fromByteData: trainingImageByteData, withHeaderInfo: trainingImageHeaderInfo) else {
    print("Could not extract training images from byte data.")
    exit(4)
}

guard let trainingLabelHeaderInfo = MNISTUtils.extractLabelHeaderInfo(fromByteData: trainingLabelByteData) else {
    print("Could not extract training label header info from byte data.")
    exit(5)
}

guard let trainingLabels = MNISTUtils.readLabels(fromByteData: trainingLabelByteData, withHeaderInfo: trainingLabelHeaderInfo) else {
    print("Could not extract training labels from byte data.")
    exit(6)
}

// EXTRACT TESTING DATA

guard let testingImageHeaderInfo = MNISTUtils.extractImageHeaderInfo(fromByteData: testingImageByteData) else {
    print("Could not extract testing images header info from byte data.")
    exit(7)
}

guard let testingImages = MNISTUtils.readImages(fromByteData: testingImageByteData, withHeaderInfo: testingImageHeaderInfo) else {
    print("Could not extract testing images from byte data.")
    exit(8)
}

guard let testingLabelHeaderInfo = MNISTUtils.extractLabelHeaderInfo(fromByteData: testingLabelByteData) else {
    print("Could not extract testing label header info from byte data.")
    exit(9)
}

guard let testingLabels = MNISTUtils.readLabels(fromByteData: testingLabelByteData, withHeaderInfo: testingLabelHeaderInfo) else {
    print("Could not extract testing labels from byte data.")
    exit(10)
}

////////////////////////////////////////////////////////////////////////////////
//                             TRAINING                                       //
////////////////////////////////////////////////////////////////////////////////
let neuralNet = NeuralNetwork()
for (index, trainingImage) in trainingImages.enumerated() {
    neuralNet.train(image: trainingImage, label: trainingLabels[index])
}

////////////////////////////////////////////////////////////////////////////////
//                             TESTING                                        //
////////////////////////////////////////////////////////////////////////////////
var numExamplesPerDigit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
var numCorrectPredictionsPerDigit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for (index, image) in testingImages.enumerated() {
    let trueLabel = Int(testingLabels[index])
    let predictedLabel = Int(neuralNet.predict(image: image))

    numExamplesPerDigit[trueLabel] += 1
    if trueLabel == predictedLabel {
        numCorrectPredictionsPerDigit[trueLabel] += 1
    }

//    print("-----------------------------------------------------------------")
//    print("               LABEL: \(testingLabels[index])          PREDICTION: \(predictedLabel)                ")
//    print("-----------------------------------------------------------------")
//    MNISTUtils.printImageGrid(image: image)
}

////////////////////////////////////////////////////////////////////////////////
//                             EVALUATION                                     //
////////////////////////////////////////////////////////////////////////////////
print("========== ACCURACY ==========")
for i in 0...9 {
    print("\(i): \((Double(numCorrectPredictionsPerDigit[i]) / Double(numExamplesPerDigit[i]))*100.0)%")
}
print("\nOVERALL: \((Double(numCorrectPredictionsPerDigit.reduce(0, +)) / Double(trainingImages.count) * 100.0))%")


