//
//  MNISTUtils.swift
//  NeuralNetMNISTClassifier
//
//  Created by WILLIAM CHAPMAN on 11/21/17.
//  Copyright © 2017 WILLIAM CHAPMAN. All rights reserved.
//

import Foundation

/**
 A collection of utility methods to read the MNIST data set.

 Format of the training dataset: http://yann.lecun.com/exdb/mnist/
 
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
 
 ***************************************************
 
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
 
 */
struct MNISTUtils {
    
    enum MNISTUtilsError: Error {
        case invalidPath(String)
    }
    
    static func readByteData(fromPath path: String) throws -> [UInt8] {
        guard let byteData = NSData(contentsOfFile: path) else {
            throw MNISTUtilsError.invalidPath("Could not find file at path: \(path)")
        }
        
        var buffer: Array<UInt8> = Array(repeating: 0, count: byteData.length)
        byteData.getBytes(&buffer, length: byteData.length)
        return buffer
    }
    
    //  MARK:  Images

    struct ImageHeaderInfo {
        let numberOfRows: UInt32
        let numberOfColumns: UInt32
        let numberOfImages: UInt32
        let magicNumber: UInt32
    }
    
    static func extractImageHeaderInfo(fromByteData byteData: [UInt8]) -> ImageHeaderInfo? {
        guard let magicNumber = UInt32(byteArray: Array(byteData[0...3])) else {
            return nil
        }
        guard let numberOfImages = UInt32(byteArray: Array(byteData[4...7])) else {
            return nil
        }
        guard let numberOfRows = UInt32(byteArray: Array(byteData[8...11])) else {
            return nil
        }
        guard let numberOfColumns = UInt32(byteArray: Array(byteData[12...15])) else {
            return nil
        }
        
        return ImageHeaderInfo(numberOfRows: numberOfRows,
                          numberOfColumns: numberOfColumns,
                          numberOfImages: numberOfImages,
                          magicNumber: magicNumber)
    }
    
    static func readImages(fromByteData byteData: [UInt8], withHeaderInfo headerInfo: ImageHeaderInfo) -> [[[Double]]]? {
        var images: [[[Double]]] = Array(repeating: Array(repeating: Array(repeating: 0.0, count: 28), count: 28), count: Int(headerInfo.numberOfImages))
        
        var imageByteCount = MemoryLayout<ImageHeaderInfo>.size
        for imageCount in 0 ..< Int(headerInfo.numberOfImages) {
            for columnNum in 0 ..< Int(headerInfo.numberOfColumns) {
                for rowNum in 0 ..< Int(headerInfo.numberOfRows)  {
                    let tempByte = byteData[imageByteCount]
                    images[imageCount][columnNum][rowNum] = Double(tempByte) / 255.0
                    imageByteCount += 1
                }
            }
        }
        
        return images
    }
    
    static func printImageGrid(image: [[Double]]) {
        for i in 0 ..< image.count {
            for j in 0 ..< image[0].count {
                if image[i][j] > 0.7 {
                    print("*", terminator: "")
                } else {
                    print(" ", terminator: "")
                }
            }
            print()
        }
    }
    
    //  MARK:  Labels

    struct LabelHeaderInfo {
        let numberOfItems: UInt32
        let magicNumber: UInt32
    }
    
    static func extractLabelHeaderInfo(fromByteData byteData: [UInt8]) -> LabelHeaderInfo? {
        guard let magicNumber = UInt32(byteArray: Array(byteData[0...3])) else {
            return nil
        }
        guard let numberOfItems = UInt32(byteArray: Array(byteData[4...7])) else {
            return nil
        }
        
        return LabelHeaderInfo(numberOfItems: numberOfItems,
                               magicNumber: magicNumber)
    }
    
    static func readLabels(fromByteData byteData: [UInt8], withHeaderInfo headerInfo: LabelHeaderInfo) -> [UInt8]? {
        return Array(byteData[MemoryLayout<LabelHeaderInfo>.size..<byteData.count])
    }
    
}

extension UInt32 {
    init?(byteArray: [UInt8]) {
        let data = Data(byteArray)
        self.init(bigEndian: data.withUnsafeBytes({ $0.pointee }))
    }
}
