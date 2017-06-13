//
//  ViewController.swift
//  CoreMLSimple
//
//  Created by 杨萧玉 on 2017/6/9.
//  Copyright © 2017年 杨萧玉. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController, UIImagePickerControllerDelegate {
    
    //let inputSize:CGFloat = 299.0
    let inputSize:CGFloat = 32.0
    
    // Outlets to label and view
    @IBOutlet private weak var predictLabel: UILabel!
    @IBOutlet private weak var previewView: UIView!
    
    // some properties used to control the app and store appropriate values
    
    let inceptionv3model = Inceptionv3()
    let cifar10model = cifar10()
    
    private var videoCapture: VideoCapture!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        let spec = VideoSpec(fps: 3, size: CGSize(width: inputSize, height: inputSize))
        videoCapture = VideoCapture(cameraType: .back,
                                    preferredSpec: spec,
                                    previewContainer: previewView.layer)
        videoCapture.imageBufferHandler = {[unowned self] (imageBuffer, timestamp, outputBuffer) in
            do {
                let image: CVPixelBuffer = self.resize(imageBuffer: imageBuffer)!
                
                let prediction = try self.cifar10model.prediction(image: image)
//                let prediction = try self.inceptionv3model.prediction(image: image)
                DispatchQueue.main.async {
                    self.predictLabel.text = prediction.classLabel
                }
            }
            catch let error as NSError {
                fatalError("Unexpected error ocurred: \(error.localizedDescription).")
            }
        }
    }
    
    func resize(imageBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        var ciImage = CIImage(cvPixelBuffer: imageBuffer, options: nil)
        let transform = CGAffineTransform(scaleX: inputSize / CGFloat(CVPixelBufferGetWidth(imageBuffer)), y: inputSize / CGFloat(CVPixelBufferGetHeight(imageBuffer)))
        ciImage = ciImage.applying(transform).cropping(to: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        let ciContext = CIContext()
        var resizeBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(inputSize), Int(inputSize), CVPixelBufferGetPixelFormatType(imageBuffer), nil, &resizeBuffer)
        ciContext.render(ciImage, to: resizeBuffer!)
        return resizeBuffer
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        guard let videoCapture = videoCapture else {return}
        videoCapture.startCapture()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        guard let videoCapture = videoCapture else {return}
        videoCapture.resizePreview()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        guard let videoCapture = videoCapture else {return}
        videoCapture.stopCapture()
        
        navigationController?.setNavigationBarHidden(false, animated: true)
        super.viewWillDisappear(animated)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
}

