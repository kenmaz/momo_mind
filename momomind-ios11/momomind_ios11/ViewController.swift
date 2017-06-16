//
//  ViewController.swift
//  momomind_ios11
//
//  Created by Kentaro Matsumae on 2017/06/16.
//  Copyright © 2017年 kenmaz.net. All rights reserved.
//

import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController {
    
    @IBOutlet weak var overlayView: OverlayView!
    
    let session = AVCaptureSession()
    var device: AVCaptureDevice?
    var previewLayer: AVCaptureVideoPreviewLayer?
    var connection : AVCaptureConnection?
    
    lazy var faceRequest: VNDetectFaceRectanglesRequest = {
        return VNDetectFaceRectanglesRequest(completionHandler: self.vnRequestHandler)
    }()
    
    var inputImage:CIImage?
    var overlayViewSize: CGSize?
    var videoDims: CMVideoDimensions?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupVideoCapture()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        overlayViewSize = overlayView.frame.size
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if !session.isRunning {
            session.startRunning()
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if session.isRunning {
           session.stopRunning()
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.layer.bounds
    }
}

extension ViewController {
    
    func setupVideoCapture() {
        let device = AVCaptureDevice.default(for: AVMediaType.video)!
        self.device = device
        session.sessionPreset = AVCaptureSession.Preset.inputPriority
        device.formats.forEach { (format) in
            print(format)
        }
        print("format:",device.activeFormat)
        print("min duration:", device.activeVideoMinFrameDuration)
        print("max duration:", device.activeVideoMaxFrameDuration)
        
        let desc = device.activeFormat.formatDescription
        self.videoDims = CMVideoFormatDescriptionGetDimensions(desc)
        
        // Input
        let deviceInput: AVCaptureDeviceInput
        do {
            deviceInput = try AVCaptureDeviceInput(device: device)
        } catch {
            fatalError()
        }
        guard session.canAddInput(deviceInput) else {
            fatalError()
        }
        session.addInput(deviceInput)
        
        // Preview:
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.frame = view.layer.bounds
        previewLayer.contentsGravity = kCAGravityResizeAspectFill
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        view.layer.insertSublayer(previewLayer, at: 0)
        self.previewLayer = previewLayer
        
        // Output
        let output = AVCaptureVideoDataOutput()
        let key = kCVPixelBufferPixelFormatTypeKey as String
        let val = kCVPixelFormatType_32BGRA as NSNumber
        output.videoSettings = [key: val]
        output.alwaysDiscardsLateVideoFrames = true
        let queue = DispatchQueue(label: "net.kenmaz.momomind")
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            fatalError()
        }
        session.addOutput(output)
        
        self.connection = output.connection(with: AVMediaType.video)
    }
    
    func vnRequestHandler(request: VNRequest, error: Error?) {
        if let e = error {
            print(e)
            return
        }
        guard let req = request as? VNDetectFaceRectanglesRequest, let faces = req.results as? [VNFaceObservation] else {
            return
        }
        guard let image = inputImage, let viewSize = overlayViewSize else {
            return
        }
        let imageSize = image.extent.size
        
        var boxes:[CGRect] = []
        faces.forEach { (face) in
            print("boundingBox:",face.boundingBox, viewSize)
            let box = face.boundingBox.scaled(to: viewSize)
            guard image.extent.contains(box) else {
                return
            }
            boxes.append(box)
        }
        DispatchQueue.main.async {
            self.overlayView.boxes = boxes
            self.overlayView.setNeedsDisplay()
        }
        
        //device = 1920,1080 (imagesize/videoDim)
        //image  = 736, 414
        //screen = 414, 736
    }
}

extension CGRect {
    func scaled(to size: CGSize) -> CGRect {
        return CGRect(
            x: self.origin.x * size.width,
            y: (1.0 - self.origin.y - self.size.height) * size.height,
            width: (self.size.width * size.width),
            height: (self.size.height * size.height)
        )
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if connection.videoOrientation != .portrait {
            connection.videoOrientation = .portrait
            return
        }
        if let buffer = CMSampleBufferGetImageBuffer(sampleBuffer), connection == self.connection {
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            processBuffer(timestamp, buffer)
        }
    }
    
    fileprivate func processBuffer(_ timestamp: CMTime, _ buffer: CVImageBuffer) {
        let inputImage = CIImage(cvImageBuffer: buffer)
        let handler = VNImageRequestHandler(ciImage: inputImage)
        self.inputImage = inputImage
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([self.faceRequest])
            } catch {
                print(error)
            }
        }
    }
}
