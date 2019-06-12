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
    
    lazy var momomind: Model = {
        let m = Model()
        return m
    }()
    
    var currentName: String = ""
    var success = 0
    
    @IBAction func trainButtonDidTap(_ sender: Any) {
        execute(type: "train")
    }
    
    @IBAction func testButtonDidTap(_ sender: Any) {
        execute(type: "test")
    }
    
    @IBAction func closeButtonDidTap(_ sender: Any) {
        dismiss(animated: true, completion: nil)
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


//MARK: debug

extension TestViewController {
    
    func dumpRawData(imageRef: CGImage) {
        guard let data = imageRef.dataProvider?.data else {
            fatalError()
        }
        let length = CFDataGetLength(data)
        var rawData = [UInt8](repeating: 0, count: length)
        let range = CFRange(location: 0, length: length)
        CFDataGetBytes(data, range, &rawData)
        print(rawData)
    }

    //FIXME: dosen't work
    func _dumpRawData(from image: CGImage) {
        
        let options = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pxBuffer: CVPixelBuffer? = nil
        let width = image.width
        let height = image.height
        let res = CVPixelBufferCreate(kCFAllocatorDefault,
                                      width,
                                      height,
                                      kCVPixelFormatType_32ARGB,
                                      options as CFDictionary?,
                                      &pxBuffer)
        guard res == kCVReturnSuccess else {
            assertionFailure()
            return
        }
        guard let buf = pxBuffer else {
            assertionFailure()
            return
        }
        let ctx = CIContext()
        let input = CIImage(cgImage: image)
        ctx.render(input, to: buf)
        
        guard CVPixelBufferLockBaseAddress(buf, CVPixelBufferLockFlags.readOnly) == kCVReturnSuccess else {
            assertionFailure()
            return
        }
        let size = CVPixelBufferGetDataSize(buf)
        print(size)
        guard let pointer = CVPixelBufferGetBaseAddress(buf) else {
            assertionFailure()
            return
        }
        let p = pointer.bindMemory(to: Int8.self, capacity: 128)
        for i in 0..<128 {
            print(p.advanced(by: i).pointee)
        }
        
        CVPixelBufferUnlockBaseAddress(buf, CVPixelBufferLockFlags.readOnly)
    }
}
