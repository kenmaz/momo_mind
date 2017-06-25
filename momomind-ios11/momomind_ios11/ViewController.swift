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
    
    func classificationRequest() -> VNCoreMLRequest {
        do {
            let model = try VNCoreMLModel(for: self.momomind.model)
            return VNCoreMLRequest(model: model, completionHandler: self.handleClassification)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }
    
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
    
    @IBAction func infoButtonDidTap(_ sender: Any) {
        eval()
    }
}

extension ViewController {
    
    func handleClassification(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let best = observations.first
            else { fatalError("can't get best result") }
        
//        print("-----")
//        print(request)
//        for ob in observations {
//            print(ob.identifier, ob.confidence)
//        }
        
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
        guard
            let req = request as? VNDetectFaceRectanglesRequest,
            let faces = req.results as? [VNFaceObservation],
            let firstFace = faces.first else {
            return
        }
        guard let image = inputImage else {
            return
        }
        
        drawFaceRectOverlay(image: image, faces: [firstFace])
        processForPredict(image: image, faces: [firstFace])
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
        let size = CGFloat(inputSize)
        
        let transform = CGAffineTransform(
            scaleX: size / box.size.width,
            y: size / box.size.height)
        let faceImage = image.cropping(to: box).applying(transform)
        
        let ctx = CIContext()
        guard let cgImage = ctx.createCGImage(faceImage, from: faceImage.extent) else {
            assertionFailure()
            return
        }
        let uiImage = UIImage(cgImage: cgImage)
        DispatchQueue.main.async {
            self.facePreview.image = uiImage
        }
        
        let predInput = CIImage(cgImage: cgImage)
        predicate_using_vision_api(image: predInput)
    }
    
    fileprivate func predicate_using_vision_api(image: CIImage) {
        print(image)
        
        let handler = VNImageRequestHandler(ciImage: image)
        do {
            let req = classificationRequest()
            try handler.perform([req])
        } catch {
            print(error)
        }
    }
    
    fileprivate func predicate(image: CIImage) {
//        let context = CIContext()
//        let cgImage = context.createCGImage(image, from: image.extent)
//        if let label = classifiy(cgImage: cgImage) {
//            DispatchQueue.main.async {
//                self.label.text = label
//            }
//        }
    }
    
    fileprivate func classifiy(cgImage: CGImage?) -> String? {
        guard let input = mlMultiArray(from: cgImage) else {
            return nil
        }
        do {
            return nil
//            let res = try momomind.prediction(image: input)
//            return res.classLabel
        } catch {
            print(error)
            return nil
        }
    }
    
    private func mlMultiArray(from cgImage: CGImage?) -> MLMultiArray? {
        
        guard let data = cgImage?.dataProvider?.data else {
            assertionFailure()
            return nil
        }
        let length = CFDataGetLength(data)
        var rawData = [UInt8](repeating: 0, count: length)
        CFDataGetBytes(data, CFRange(location: 0, length: length), &rawData)
        
        let size = NSNumber(value: inputSize)
        
        guard let input = try? MLMultiArray(shape: [3, size, size], dataType: MLMultiArrayDataType.float32) else {
            assertionFailure()
            return nil
        }
        var index = 0
        var outIdx = 0
        
        let len = rawData.count
        while index < len {
            let r = rawData[index + 0]
            let g = rawData[index + 1]
            let b = rawData[index + 2]
            let a = rawData[index + 3]
            let m:Float = 255.0
            input[outIdx + 0] = NSNumber(value: Float(r) / m)
            input[outIdx + 1] = NSNumber(value: Float(g) / m)
            input[outIdx + 2] = NSNumber(value: Float(b) / m)
            index += 4
            outIdx += 3
        }
//
//        for (i, data) in rawData.enumerated() {
//            if i % 4 == 3 {
//                continue //alpha
//            } else {
//                let norm = Float(data) / 255.0
//                input[index] = NSNumber(value: norm)
//                index += 1
//            }
//        }
        return input
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


//eval

extension ViewController {
    
    func eval() {
        let names = [
//            "reni",
//            "kanako",
//            "shiori",
//            "arin",
            "momoka"
        ]
        for memberName in names {
            print(memberName)
            let jpgs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: "out/train/\(memberName)")
            for path in jpgs {
                let image = UIImage(contentsOfFile: path)?.cgImage
                let input = CIImage(cgImage: image!)
                predicate_using_vision_api(image: input)
            }
        }
    }
    
    func ___eval() {
        let path = Bundle.main.path(forResource: "0_c110ce436ca73d6c70fb2af646563c60", ofType: "jpg", inDirectory: "out/train/arin")!
        let image = UIImage(contentsOfFile: path)?.cgImage
        let input = CIImage(cgImage: image!)
        predicate_using_vision_api(image: input)
    }
    
    func __eval() {
        let path = Bundle.main.path(forResource: "0_c110ce436ca73d6c70fb2af646563c60", ofType: "jpg", inDirectory: "out/train/arin")!
        //let path = Bundle.main.path(forResource: "test", ofType: "png", inDirectory: "out/train/arin")!
        let image = UIImage(contentsOfFile: path)
        let label = classifiy(cgImage: image?.cgImage)!
        print(label)
    }
    
    func _eval() {
        print("processing..")
        DispatchQueue.global().async {
            self.evalInBg()
            print("done")
        }
    }
    
    func evalInBg() {
        let names = [
            "reni",
            "kanako",
            "shiori",
            "arin",
            "momoka"
        ]
        for memberName in names {
            print(memberName)
            let jpgs = Bundle.main.paths(forResourcesOfType: "jpg", inDirectory: "out/train/\(memberName)")
            var errorCnt:Float = 0
            var successCnt:Float = 0
            let total = Float(jpgs.count)
            
            for path in jpgs {
                let image = UIImage(contentsOfFile: path)
                if let label = classifiy(cgImage: image?.cgImage) {
                    if memberName == label {
                        successCnt += 1
                    }
                } else {
                    errorCnt += 1
                }
            }
            print("result success:\(successCnt) / \(total) ( \(successCnt / total) %),  err:\(errorCnt)")
        }
    }
    
}
