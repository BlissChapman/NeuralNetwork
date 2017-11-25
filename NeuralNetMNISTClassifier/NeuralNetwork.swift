//
//  NeuralNetwork.swift
//  NeuralNetMNISTClassifier
//
//  Created by Bliss Chapman on 11/24/17.
//  Copyright © 2017 WILLIAM CHAPMAN. All rights reserved.
//

import Foundation

struct NeuralNetwork {

    private var weights: [[[Double]]] = []
    let learningRate: Double

    init(layerWidths: [UInt], learningRate: Double = 0.1) {
        self.learningRate = learningRate

        for (layerIdx, layerWidth) in layerWidths[0 ..< layerWidths.endIndex-1].enumerated() {
            let nextLayerWidth = layerWidths[layerIdx + 1]

            var randomWeightsLayer: [[Double]] = []

            for _ in 0 ..< layerWidth {
                var randomNodeWeights: [Double] = []
                for _ in 0 ..< nextLayerWidth {
                    randomNodeWeights.append(Double(arc4random_uniform(100000)) / 100000.0)
                }
                randomWeightsLayer.append(randomNodeWeights)
            }

            weights.append(randomWeightsLayer)
        }
    }

    mutating func train(input x: [Double], desiredOutput y: [Double]) -> [Double] {
        guard x.count == weights.first?.count else {
            print("Cannot train on input of invalid size.  Input size: \(x.count).  Input layer width: \(weights[0].count)")
            return []
        }

        // Forward Propagation
        let inferenceResults = inference(input: x)

        guard let predictions = inferenceResults.perceptronActivations.last else {
            print("Unknown inference failure 😥")
            return []
        }

        // Backward Propagation
        // Initialize delta array.
        var delta = [[Double]]()
        for l in 0..<inferenceResults.perceptronInputs.count {
            let delta_l = Array<Double>(repeatElement(0.0, count: inferenceResults.perceptronInputs[l].count))
            delta.append(delta_l)
        }

        // Compute the ∆ values for the output units, using the observed error.
        let outputLayerIdx = inferenceResults.perceptronInputs.count - 1
        for j in 0..<inferenceResults.perceptronInputs[outputLayerIdx].count {
            let error = y[j] - inferenceResults.perceptronActivations[outputLayerIdx][j]
            delta[outputLayerIdx][j] = sigmoidDerivative(inferenceResults.perceptronInputs[outputLayerIdx][j]) * error
        }

        // Propagate deltas backward from output layer to input layer.
        for l in (0..<weights.count).reversed() {
            for i in 0..<weights[l].count {
                var sum = 0.0
                for j in 0..<weights[l][i].count {
                    sum += (weights[l][i][j] * delta[l + 1][j])
                }
                delta[l][i] = sigmoidDerivative(inferenceResults.perceptronInputs[l][i]) * sum
            }
        }

        // Update every weight in network using deltas.
        for l in 0..<weights.count {
            for i in 0..<weights[l].count {
                for j in 0..<weights[l][i].count {
                    weights[l][i][j] += (learningRate * inferenceResults.perceptronActivations[l][i] * delta[l][j])
                }
            }
        }

        return predictions
    }

    func predict(fromTestInput input: [Double]) -> [Double] {
        guard input.count == weights.first?.count else {
            print("Cannot predict for input of invalid size.  Input size: \(input.count).  Input layer width: \(weights[0].count)")
            return []
        }

        guard let predictions = inference(input: input).perceptronActivations.last else {
            print("Unknown inference failure 😥")
            return []
        }

        return predictions
    }

    private func inference(input: [Double]) -> (perceptronInputs: [[Double]], perceptronActivations: [[Double]]) {
        var perceptronActivations = [[Double]]()
        perceptronActivations.append(input)

        var perceptronInputs = [[Double]]()
        perceptronInputs.append(input)

        for l in 0..<weights.count {
            var nextLayer = [Double]()

            for i in 0..<weights[l].count {
                if nextLayer.count == 0 {
                    nextLayer = Array(repeatElement(0.0, count: weights[l][i].count))
                }
                for j in 0..<weights[l][i].count {
                    nextLayer[j] += (weights[l][i][j] * perceptronActivations[l][i])
                }
            }

            perceptronInputs.append(nextLayer)
            perceptronActivations.append(nextLayer.map({ sigmoid($0) }))
        }

        return (perceptronInputs, perceptronActivations)
    }

    private func sigmoid(_ z: Double) -> Double {
        return 1.0 / (1.0 + exp(-z))
    }

    private func sigmoidDerivative(_ z: Double) -> Double {
        let x = sigmoid(z)
        return x*(1.0-x)
    }
}
