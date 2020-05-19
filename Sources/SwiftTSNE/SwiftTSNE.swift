import Foundation
import os.log

public struct Tsne {
    
    public init() {}
    
    let log = OSLog(subsystem: "SwiftTSNE", category: String(describing: "Tsne"))
    
    public func transform(data: [[Double]],
                          maxIterations: Int = 300,
                          learningRate: Double = 0.01,
                          completion: @escaping ([[Double]]) -> Void) {
        let queue = DispatchQueue(label: "processing",
                                  qos: .userInitiated)
        queue.async {
            let result = self.transform(data: data, maxIterations: maxIterations, learningRate: learningRate)
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }
    
    public func transform(data: [[Double]], maxIterations: Int = 300, learningRate: Double = 0.01) -> [[Double]] {
        
        // step 1: compute pairwise affinities (not controlling perplexity though)
        
        var probabilities = [[Double]]()
        
        for i in 0..<data.count {
            for j in 0..<data.count {
                let std = 2 * pow(data[i].std(), 2)
                let num = exp(-differenceSquared(lhs: data[i], rhs: data[j])/(std))
                let den = data.enumerated()
                    .filter { $0.offset != i }
                    .map { exp(-differenceSquared(lhs: $0.element, rhs: data[i])/(std)) }
                    .reduce(0, +)
                if j == 0 {
                    if i == 0 {
                        probabilities = [[num/den]]
                    } else {
                        probabilities.append([num/den])
                    }
                } else {
                    probabilities[i].append(num/den)
                }
            }
        }
        
        for i in 0..<probabilities.count {
            for j in 0..<probabilities[i].count {
                let avg = (probabilities[i][j] + probabilities[j][i]) / (2 * Double(probabilities[i].count))
                probabilities[i][j] = avg
                probabilities[j][i] = avg
                if i == j {
                    probabilities[i][j] = 0
                }
            }
        }
        
        for i in 0..<probabilities.count {
            let sum = probabilities[i].reduce(0.0, +)
            for j in 0..<probabilities.count {
                probabilities[i][j] = probabilities[i][j]/sum
            }
        }
        
        // step 2: creating random points in destiny space
        var result = data.map { _ in [Double(arc4random_uniform(1000)) / 1000.0,  Double(arc4random_uniform(1000)) / 1000.0] }
        
        for _ in 0..<maxIterations {
            var generatedDistances = [[Double]]()
            
            for i in 0..<result.count {
                for j in 0..<result.count {
                    let diff = differenceSquared(lhs: result[i], rhs: result[j])
                    let num = pow((1 + diff), -1)
                    let den = result.enumerated()
                        .filter { $0.offset != i }
                        .map { pow(1 + differenceSquared(lhs: result[$0.offset], rhs: result[i]), -1) }
                        .reduce(0, +)
                    if j == 0 {
                        if i == 0 {
                            generatedDistances = [[num/den]]
                        } else {
                            generatedDistances.append([num/den])
                        }
                    } else {
                        generatedDistances[i].append(num/den)
                    }
                }
                
            }
            
            // step 3: calculate error
            var error = 0.0
            
            for i in 0..<probabilities.count {
                for j in 0..<probabilities.count {
                    if i != j {
                        error += probabilities[i][j] * Foundation.log(probabilities[i][j]/generatedDistances[i][j])
                    }
                }
            }
            
            os_log(.info, log: log, "Kullback Leibler divergence: %{PUBLIC}@", "\(error)")
            
            // step 4: update result
            var updates = [[Double]]()
            for i in 0..<result.count {
                if result[i].count != 2 {
                    print(i)
                    print(result)
                }
                var sum = [Double]()
                for j in 0..<result.count {
                    if j == 0 {
                        sum = result[i].map { _ in 0.0 }
                    }
                    if i == j { continue }
                    let a = (probabilities[i][j] - generatedDistances[i][j])
                    let b = pow(1 + differenceSquared(lhs: result[i], rhs: result[j]), -1)
                    let partialSum = multiply(scalar: a * b, vector: subtract(lhs: result[i], rhs: result[j]))
                    sum = add(lhs: sum, rhs: partialSum)
                }
                sum = multiply(scalar: 4, vector: sum)
                updates.append(sum)
            }
            
            result = zip(result, updates).map {
                return subtract(lhs: $0.0, rhs: multiply(scalar: learningRate, vector: $0.1))
            }
        }
        
        return result
    }

}
