//
//  NeuralNetwork.swift
//  NeuralNetMNISTClassifier
//
//  Created by Bliss Chapman & William Chapman on 11/24/17.
//  Copyright © 2017 Bliss Chapman & William Chapman. All rights reserved.
//

import Foundation
import Accelerate

// =============================================================================
//                             Custom Operators
// =============================================================================

precedencegroup DotProductPrecedence {
    lowerThan: AdditionPrecedence
    associativity: left
}

infix operator •: DotProductPrecedence

func •(a: [Double], b: [Double]) -> Double {
    var result = 0.0
    vDSP_dotprD(a, 1, b, 1, &result, vDSP_Length(a.count))
    return result
}

// =============================================================================
// =============================================================================


/**
 */
struct NeuralNetwork {

    private let learningRate: Double
    private var weights: [[[Double]]] = []

    init(layerWidths: [UInt], learningRate: Double) {
        self.learningRate = learningRate
        initializeRandomWeights(withLayerWidths: layerWidths)
    }

    private mutating func initializeRandomWeights(withLayerWidths layerWidths: [UInt]) {
        for layerIdx in 0..<(layerWidths.endIndex-1) {
            let nextLayerWidth = layerWidths[layerIdx + 1]
            var randomWeightsLayer: [[Double]] = []

            for _ in 0..<layerWidths[layerIdx] {
                var randomNodeWeights: [Double] = []
                for _ in 0 ..< nextLayerWidth {
                    let randomInt = randomValue(inRange: -10000..<10000)
                    randomNodeWeights.append(Double(randomInt) / 10000.0)
                }
                randomWeightsLayer.append(randomNodeWeights)
            }

            self.weights.append(randomWeightsLayer)
        }
    }

    mutating func train(input x: [Double], desiredOutput y: [Double]) -> [Double] {
        guard x.count == weights.first?.count else {
            print("Cannot train on input of invalid size.  Input size: \(x.count).  Input layer width: \(weights[0].count)")
            return []
        }

        // Forward Propagation
        let perceptronActivations = inference(input: x)
        guard let predictions = perceptronActivations.last else {
            print("Unknown inference failure 😥")
            return []
        }

        // Backward Propagation
        // Initialize delta array.
        var delta = [[Double]]()
        for l in 0..<perceptronActivations.count {
            delta.append(Array(repeatElement(0.0, count: perceptronActivations[l].count)))
        }

        // Compute the ∆ values for the output units, using the observed error.
        let outputLayerIdx = perceptronActivations.count - 1
        for j in 0..<perceptronActivations[outputLayerIdx].count {
            let error = y[j] - perceptronActivations[outputLayerIdx][j]
            delta[outputLayerIdx][j] = sigmoidDerivative(perceptronActivations[outputLayerIdx][j]) * error
        }

        // Propagate deltas backward from output layer to input layer.
        for l in (0..<weights.count).reversed() {
            for i in 0..<weights[l].count {
                let sumLayerError = weights[l][i] • delta[l + 1]
                delta[l][i] = sigmoidDerivative(perceptronActivations[l][i]) * sumLayerError
            }
        }

        // Update every weight in network using deltas.
        for l in 0..<weights.count {
            for i in 0..<weights[l].count {
                for j in 0..<weights[l][i].count {
                    weights[l][i][j] += (learningRate * perceptronActivations[l][i] * delta[l + 1][j])
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

        guard let predictions = inference(input: input).last else {
            print("Unknown inference failure 😥")
            return []
        }

        return predictions
    }

    private func inference(input: [Double]) -> [[Double]] {
        var perceptronActivations = [[Double]]()
        perceptronActivations.append(input)

        for l in 0..<weights.count {
            let numPerceptronsInNextLayer = weights[l][0].count
            perceptronActivations.append(Array(repeatElement(0.0, count: numPerceptronsInNextLayer)))

            for i in 0..<weights[l].count {
                for j in 0..<weights[l][i].count {
                    perceptronActivations[l + 1][j] += (weights[l][i][j] * perceptronActivations[l][i])
                }
            }

            perceptronActivations[l + 1] = perceptronActivations[l + 1].map({ sigmoid($0) })
        }

        return perceptronActivations
    }

    // Activation Functions

    private func relu(_ z: Double) -> Double {
      return max(Double.leastNonzeroMagnitude, z)
    }

    private func reluDerivative(_ z: Double) -> Double {
      if z < 0 {
        return 0.0
      } else {
        return 1.0
      }
    }

    private func sigmoid(_ z: Double) -> Double {
        return 1.0 / (1.0 + exp(-z))
    }

    private func sigmoidDerivative(_ z: Double) -> Double {
        let x = sigmoid(z)
        return x*(1.0-x)
    }

    private func randomValue(inRange range: Range<Int>) -> Int {
      return Int(arc4random_uniform(UInt32(range.upperBound - range.lowerBound))) + range.lowerBound
    }
}
