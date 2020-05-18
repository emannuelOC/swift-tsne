import XCTest
import NaturalLanguage
@testable import SwiftTSNE

final class SwiftTSNETests: XCTestCase {
        
    func testDistances() {
        let tsne = Tsne()
        
        let data = prepareTestData()
        
        let results = tsne.transform(data: data.0)
        
        let distanceDogCat = calculateDistance(lhs: results[0], rhs: results[1])
        let distanceCatMom = calculateDistance(lhs: results[1], rhs: results[2])
        let distanceMomDad = calculateDistance(lhs: results[2], rhs: results[3])
        
        XCTAssert(distanceDogCat < distanceCatMom, "Distance between dog-cat should be less than cat-mom")
        XCTAssert(distanceCatMom > distanceMomDad, "Distance between cat-mom should be greater than mom-dad")
    }

    static var allTests = [
        ("testDistances", testDistances),
    ]
    
    func prepareTestData() -> ([[Double]], [Double]) {
        let text = "cachorro gato mãe pai"
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

        let vectors = tokens.compactMap { ($0, embedding?.vector(for: $0)) }
        
        return (
            vectors.compactMap { $0.1 },
            distances
        )
    }
    
    func calculateDistance(lhs: [Double], rhs: [Double]) -> Double {
        return sqrt(differenceSquared(lhs: lhs, rhs: rhs))
    }
}
