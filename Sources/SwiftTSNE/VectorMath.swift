//
//  File.swift
//  
//
//  Created by Emannuel Carvalho on 18/05/20.
//

import Foundation

func differenceSquared(lhs: [Double], rhs: [Double]) -> Double {
    if lhs.count != rhs.count {
        return 0
    }
    
    return zip(lhs, rhs)
        .map { pow(($0 - $1), 2) }
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

