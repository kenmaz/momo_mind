//
//  TestViewController.swift
//  momomind_ios11
//
//  Created by Kentaro Matsumae on 2017/06/25.
//  Copyright © 2017年 kenmaz.net. All rights reserved.
//

import UIKit
import Vision

class TestViewController: UIViewController {
    
    let momomind = Momomind()
    
    var currentName: String = ""
    var success = 0
    
    @IBAction func trainButtonDidTap(_ sender: Any) {
        execute(type: "train")
    }
    
    @IBAction func testButtonDidTap(_ sender: Any) {
        execute(type: "test")
    }
    
    func execute(type: String) {
        let names = [
            "reni",
            "kanako",
            "shiori",
            "arin",
            "momoka"
        ]
        var total = 0
        success = 0
        
        for name in names {
            print("processing..\(type) \(name)")
            
            currentName = name
            let dir = "out/\(type)/\(name)"
            let jpgs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: dir)
            total += jpgs.count
            
            for path in jpgs {
                let image = UIImage(contentsOfFile: path)?.cgImage
                let input = CIImage(cgImage: image!)
                predicate(image: input)
            }
        }
        let acc = Float(success) / Float(total)
        print("success: \(success) / \(total), accu:\( acc )")
    }
    
    private func predicate(image: CIImage) {
        let handler = VNImageRequestHandler(ciImage: image)
        do {
            let model = try VNCoreMLModel(for: self.momomind.model)
            let req = VNCoreMLRequest(model: model, completionHandler: self.handleClassification)
            try handler.perform([req])
        } catch {
            print(error)
        }
    }
    
    private func handleClassification(request: VNRequest, error: Error?) {
        guard
            let observations = request.results as? [VNClassificationObservation],
            let best = observations.first else {
            fatalError("unexpected result type from VNCoreMLRequest")
        }
        if currentName == best.identifier {
            success += 1
        }
    }
    
}
