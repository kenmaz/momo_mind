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
    
    @IBOutlet var progressViews: [UIProgressView]! {
        didSet {
            progressViews.forEach {
                $0.transform = CGAffineTransform(scaleX: 1, y: 3)
                $0.progress = 0
            }
        }
    }
    
    @IBOutlet var probabilityLabels: [UILabel]! {
        didSet {
            probabilityLabels.forEach {
                $0.text = "-%"
            }
        }
    }
    
    let session = AVCaptureSession()
    var device: AVCaptureDevice?
    var previewLayer: AVCaptureVideoPreviewLayer?
    var connection : AVCaptureConnection?
    let inputSize: Float = 112
    
    let momomind = Momomind()
    
    lazy var faceRequest: VNDetectFaceRectanglesRequest = {
        return VNDetectFaceRectanglesRequest(completionHandler: self.vnRequestHandler)
    }()
    
    var inputImage:CIImage?
    var overlayViewSize: CGSize?
    var videoDims: CMVideoDimensions?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard TARGET_OS_SIMULATOR != 1 else {
            print("simulator!!!")
            return
        }
        
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

//MARK: - setup video

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
}

//MARK: - Video Capture

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
    
    private func processBuffer(_ timestamp: CMTime, _ buffer: CVImageBuffer) {
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

//MARK: - Face detection

extension ViewController {
    
    func vnRequestHandler(request: VNRequest, error: Error?) {
        if let e = error {
            print(e)
            return
        }
        guard
            let req = request as? VNDetectFaceRectanglesRequest,
            let faces = req.results as? [VNFaceObservation],
            let centerFace = faces.sorted(by: { (a, b) -> Bool in distanceToCenter(face: a) < distanceToCenter(face: b) }).first else {
            return
        }
        guard let image = inputImage else {
            return
        }
        
        drawFaceRectOverlay(image: image, face: centerFace)
        
        guard let cgImage = getFaceCGImage(image: image, face: centerFace) else {
            return
        }
        showPreview(cgImage: cgImage)
        predicate(cgImage: cgImage)
    }
    
    private func distanceToCenter(face: VNFaceObservation) -> CGFloat {
        let x = face.boundingBox.origin.x + face.boundingBox.size.width / 2
        let y = face.boundingBox.origin.y + face.boundingBox.size.height / 2
        let pos = CGPoint(x: x, y: y)
        let viewPos = CGPoint(x: 0.5, y: 0.5)
        let distance = sqrt(pow(pos.x - viewPos.x, 2) + pow(pos.y - viewPos.y, 2))
        return distance
    }
    
    //device = 1920,1080 (imagesize/videoDim)
    //image  = 736, 414
    //screen = 414, 736
    //
    private func drawFaceRectOverlay(image: CIImage, face: VNFaceObservation) {
        guard let viewSize = overlayViewSize else {
            return
        }
        
        var boxes:[CGRect] = []
        print("boundingBox:",face.boundingBox, viewSize)
        let box = face.boundingBox.scaledForOverlay(to: viewSize)
        boxes.append(box)
        
        DispatchQueue.main.async {
            self.overlayView.boxes = boxes
            self.overlayView.setNeedsDisplay()
        }
    }
    
    private func getFaceCGImage(image: CIImage, face: VNFaceObservation) -> CGImage? {
        let imageSize = image.extent.size
        
        let box = face.boundingBox.scaledForCropping(to: imageSize)
        guard image.extent.contains(box) else {
            return nil
        }
        let size = CGFloat(inputSize)
        
        let transform = CGAffineTransform(
            scaleX: size / box.size.width,
            y: size / box.size.height)
        let faceImage = image.cropping(to: box).applying(transform)
        
        let ctx = CIContext()
        guard let cgImage = ctx.createCGImage(faceImage, from: faceImage.extent) else {
            assertionFailure()
            return nil
        }
        return cgImage
    }
    
    private func showPreview(cgImage:CGImage) {
        let uiImage = UIImage(cgImage: cgImage)
        DispatchQueue.main.async {
            self.facePreview.image = uiImage
        }
    }
}

//MARK: - Predicate

extension ViewController {
    
    fileprivate func predicate(cgImage: CGImage) {
        let image = CIImage(cgImage: cgImage)
        
        print(image)
        
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
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        
        DispatchQueue.main.async {
            for ob in observations {
                switch ob.identifier {
                case "reni": self.updateLabel(idx: 0, ob: ob)
                case "kanako": self.updateLabel(idx: 1, ob: ob)
                case "shiori": self.updateLabel(idx: 2, ob: ob)
                case "arin": self.updateLabel(idx: 3, ob: ob)
                case "momoka": self.updateLabel(idx: 4, ob: ob)
                default:
                    break
                }
            }
        }
    }
    
    private func updateLabel(idx: Int, ob: VNClassificationObservation) {
        let label = probabilityLabels.filter{ $0.tag == idx }.first
        let progress = progressViews.filter{ $0.tag == idx }.first
        
        let per = Int(ob.confidence * 100)
        label?.text = "\(per)%"
        progress?.progress = ob.confidence
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

