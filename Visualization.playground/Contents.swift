import NaturalLanguage

var text = "cachorro gato folha árvore galho mãe pai"
print(text)

let embedding = NLEmbedding.wordEmbedding(for: .portuguese)

func retrieveTokens(text: String) -> [String] {
    let tokenizer = NLTokenizer(unit: .word)
    tokenizer.setLanguage(.portuguese)
    tokenizer.string = text
    
    var words = [String]()
    tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { (tokenRange, _) -> Bool in
        words.append(String(text[tokenRange].lowercased()))
        return true
    }
    return words
}

let tokens = retrieveTokens(text: text)

let distances: [Double] = tokens.enumerated().map { index, item in
    if index == 0 {
        return 0.0
    }
    return embedding?.distance(between: item, and: tokens[index - 1]) ?? 0.0
}

print(distances)

let vectors = tokens.compactMap { ($0, embedding?.vector(for: $0)) }

// step 1: compute pairwise affinities
let data = vectors.compactMap { $0.1 }

func differenceSquared(lhs: [Double], rhs: [Double]) -> Double {
    if lhs.count != rhs.count {
        return 0
    }
    
    return zip(lhs, rhs)
        .map { ($0 - $1) * ($0 - $1) }
        .reduce(0, +)
}

extension Array where Element: FloatingPoint {

    func sum() -> Element {
        return self.reduce(0, +)
    }

    func avg() -> Element {
        return self.sum() / Element(self.count)
    }

    func std() -> Element {
        let mean = self.avg()
        let v = self.reduce(0, { $0 + ($1-mean)*($1-mean) })
        return sqrt(v / (Element(self.count) - 1))
    }

}


func multiply(scalar: Double, vector: [Double]) -> [Double] {
    return vector.map { $0 * scalar }
}

func subtract(lhs: [Double], rhs: [Double]) -> [Double] {
    return zip(lhs, rhs).map { $0.0 - $0.1 }
}

func add(lhs: [Double], rhs: [Double]) -> [Double] {
    return zip(lhs, rhs).map { $0.0 + $0.1 }
}

func tsne(data: [[Double]], maxIterations: Int = 300, learningRate: Double = 0.01) -> [[Double]] {
    
    // step 1 (not controlling perplexity though)
    
    var probabilities = [[Double]]()
    
    for i in 0..<data.count {
        for j in 0..<data.count {
            let std = 2 * data[i].std()
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
    
    print("Step 1 - probabilities:")
    for p in probabilities {
        print(p)
    }
    
    // step 2
    var result = data.map { _ in [Double(arc4random_uniform(1000)) / 1000.0,  Double(arc4random_uniform(1000)) / 1000.0] }
    print(result)
    print("--------")
    
    for _ in 0..<maxIterations {
        print("partial result")
        print(result)
        print("--------")
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
        
        // step 3 (calculate error)
        var error = 0.0
        
        for i in 0..<probabilities.count {
            for j in 0..<probabilities.count {
                if i != j {
                    error += probabilities[i][j] * log(probabilities[i][j]/generatedDistances[i][j])
                }
            }
        }
        
        print(error)
        
        // step 4 (update result)
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
            return add(lhs: $0.0, rhs: multiply(scalar: learningRate, vector: $0.1))
        }
    }
    
    
    return result
    
}

let result = tsne(data: data)

import UIKit

extension CGPoint {

    static func from(array: [Double]) -> CGPoint {
        if array.count != 2 {
            return .zero
        }
        return CGPoint(x: array[0], y: array[1])
    }

}


let points = result
    .map { CGPoint.from(array: $0) }


func plot(points: [CGPoint], labels: [String]) -> UIView {
    let view = UIView(frame: CGRect(x: 0, y: 0, width: 500, height: 500))

    let minX = points.map { $0.x }.min() ?? 0.0
    let minY = points.map { $0.y }.min() ?? 0.0
    let maxX = points.map { $0.x }.max() ?? 0.0
    let maxY = points.map { $0.y }.max() ?? 0.0

    let virtualWidth = (maxX - minX) * 1.1
    let virtualHeight = (maxY - minY) * 1.1

    func convert(point: CGPoint) -> CGPoint {
        let x = point.x - minX
        let y = point.y - minY
        return CGPoint(x: (500 * x) / virtualWidth,
                       y: (500 * y) / virtualHeight)
    }

    for point in zip(points, labels) {
        let label = UILabel(frame: CGRect(origin: convert(point: point.0), size: CGSize(width: 50, height: 20)))
        label.font = UIFont.systemFont(ofSize: 8)
        label.text = point.1
        view.addSubview(label)
    }

    return view
}

let v = plot(points: points, labels: vectors.map { $0.0 })

import PlaygroundSupport

PlaygroundPage.current.liveView = v

