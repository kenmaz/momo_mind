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
    @IBOutlet weak var facePreview: UIImageView!
    @IBOutlet weak var label: UILabel!
    
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
    
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: Momomind().model)
            return VNCoreMLRequest(model: model, completionHandler: self.handleClassification)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    func handleClassification(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let best = observations.first
            else { fatalError("can't get best result") }
        
        DispatchQueue.main.async {
            let res = "Classification: \"\(best.identifier)\" Confidence: \(best.confidence)"
            print(res)
            self.label.text = res
        }
    }
    
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
        
        do {
            try device.lockForConfiguration()
        } catch {
            fatalError()
        }
        device.activeVideoMaxFrameDuration = CMTimeMake(1, 3)
        device.activeVideoMinFrameDuration = CMTimeMake(1, 3)
        device.unlockForConfiguration()
        
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
        guard let image = inputImage else {
            return
        }
        
        drawFaceRectOverlay(image: image, faces: faces)
        processForPredict(image: image, faces: faces)
    }
    
    //device = 1920,1080 (imagesize/videoDim)
    //image  = 736, 414
    //screen = 414, 736
    //
    fileprivate func drawFaceRectOverlay(image: CIImage, faces: [VNFaceObservation]) {
        guard let viewSize = overlayViewSize else {
            return
        }
        
        var boxes:[CGRect] = []
        faces.forEach { (face) in
            print("boundingBox:",face.boundingBox, viewSize)
            let box = face.boundingBox.scaledForOverlay(to: viewSize)
            boxes.append(box)
        }
        DispatchQueue.main.async {
            self.overlayView.boxes = boxes
            self.overlayView.setNeedsDisplay()
        }
    }
    
    fileprivate func processForPredict(image: CIImage, faces: [VNFaceObservation]) {
        let imageSize = image.extent.size
        guard let imageBuffer = image.pixelBuffer else {
            return
        }
        let type = CVPixelBufferGetPixelFormatType(imageBuffer)
        
        for face in faces {
            let box = face.boundingBox.scaledForCropping(to: imageSize)
            guard image.extent.contains(box) else {
                return
            }
            renderPreview(box: box, image: image, type: type)
        }
    }
    
    fileprivate func renderPreview(box:CGRect, image: CIImage, type: OSType) {
        let inputSize: CGFloat = 112
        let transform = CGAffineTransform(
            scaleX: inputSize / box.size.width,
            y: inputSize / box.size.height)
        let faceImage = image.cropping(to: box).applying(transform)
        
        predicate(image: faceImage)
        
        let ctx = CIContext()
        guard let cgImage = ctx.createCGImage(faceImage, from: faceImage.extent) else {
            assertionFailure()
            return
        }
        let uiImage = UIImage(cgImage: cgImage)
        DispatchQueue.main.async {
            self.facePreview.image = uiImage
        }
    }
    
    fileprivate func predicate(image: CIImage) {
        let handler = VNImageRequestHandler(ciImage: image)
        do {
            try handler.perform([classificationRequest])
        } catch {
            print(error)
        }
    }
}

extension CGRect {
    func scaledForOverlay(to size: CGSize) -> CGRect {
        return CGRect(
            x: self.origin.x * size.width,
            y: (1.0 - self.origin.y - self.size.height) * size.height,
            width: (self.size.width * size.width),
            height: (self.size.height * size.height)
        )
    }
    
    func scaledForCropping(to size: CGSize) -> CGRect {
        return CGRect(
            x: self.origin.x * size.width,
            y: self.origin.y * size.height,
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
