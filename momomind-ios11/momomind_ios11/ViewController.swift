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

    @IBAction func infoButtonDidTap(_ sender: Any) {
        let con = UIStoryboard(name: "TestViewController", bundle: nil).instantiateInitialViewController() as! TestViewController
        present(con, animated: true, completion: nil)
    }
    

    let session = AVCaptureSession()
    var device: AVCaptureDevice?
    var previewLayer: AVCaptureVideoPreviewLayer?
    var connection : AVCaptureConnection?
    let inputSize: Float = 112
    
    let momomind = Model()
    
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
        startSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopSession()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.layer.bounds
    }
}

//MARK: - setup video

extension ViewController {

    private var isActualDevice: Bool {
        return TARGET_OS_SIMULATOR != 1
    }
    
    private func startSession() {
        guard isActualDevice else { return }
        if !session.isRunning {
            session.startRunning()
        }
    }
    
    private func stopSession() {
        guard isActualDevice else { return }
        if session.isRunning {
            session.stopRunning()
        }
    }

    func setupVideoCapture() {
        guard isActualDevice else { return }

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
        guard connection == self.connection else {
            return
        }

        guard let buffer: CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let image = CIImage(cvImageBuffer: buffer)

        let cropped = resize(image: image)

        guard let croppedBuffer = pixelBufferFromImage(ciimage: cropped) else {
            return
        }

        processBuffer(buffer: croppedBuffer)
    }

    func resize(image: CIImage) -> CIImage{
        let expectedSize = CGSize(width: 416.0, height: 416.0)

        let width = image.extent.width
        let height = image.extent.height

        let widthScale = expectedSize.width / width
        let heightScale = expectedSize.height / height
        let scale = max(widthScale, heightScale)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        print(scaled.extent.width)
        print(scaled.extent.height)

        let size = CGRect(
            x: (scaled.extent.width / 2) - (expectedSize.width / 2),
            y: (scaled.extent.height / 2) - (expectedSize.height / 2),
            width: expectedSize.width,
            height: expectedSize.height
        )
        let cropped = scaled.cropped(to: size)
        return cropped
    }

    func pixelBufferFromImage(ciimage: CIImage) -> CVPixelBuffer? {
        let tmpcontext = CIContext(options: nil)
        guard let cgimage = tmpcontext.createCGImage(ciimage, from: ciimage.extent) else {
            return nil
        }

        let cfnumPointer = UnsafeMutablePointer<UnsafeRawPointer>.allocate(capacity: 1)
        let cfnum = CFNumberCreate(kCFAllocatorDefault, .intType, cfnumPointer)
        let keys: [CFString] = [kCVPixelBufferCGImageCompatibilityKey, kCVPixelBufferCGBitmapContextCompatibilityKey, kCVPixelBufferBytesPerRowAlignmentKey]
        let values: [CFTypeRef] = [kCFBooleanTrue, kCFBooleanTrue, cfnum!]
        let keysPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
        let valuesPointer =  UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 1)
        keysPointer.initialize(to: keys)
        valuesPointer.initialize(to: values)

        let options = CFDictionaryCreate(kCFAllocatorDefault, keysPointer, valuesPointer, keys.count, nil, nil)

        let width = cgimage.width
        let height = cgimage.height

        var pxbuffer: CVPixelBuffer?

        guard CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA,
            options,
            &pxbuffer) == kCVReturnSuccess else {
                return nil
        }
        
        guard CVPixelBufferLockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0)) == kCVReturnSuccess else {
            return nil
        }

        let bufferAddress = CVPixelBufferGetBaseAddress(pxbuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesperrow = CVPixelBufferGetBytesPerRow(pxbuffer!)
        guard let context = CGContext(
            data: bufferAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesperrow,
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue) else {
                return nil
        }

        context.concatenate(CGAffineTransform(rotationAngle: 0))
        context.concatenate(__CGAffineTransformMake( 1, 0, 0, -1, 0, CGFloat(height) )) //Flip Vertical

        context.draw(cgimage, in: CGRect(x:0, y:0, width:CGFloat(width), height:CGFloat(height)))
        guard CVPixelBufferUnlockBaseAddress(pxbuffer!, CVPixelBufferLockFlags(rawValue: 0)) == kCVReturnSuccess else {
            return nil
        }
        return pxbuffer!

    }

    func getImageFromSampleBuffer (buffer:CMSampleBuffer) -> UIImage? {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let resizedCIImage = ciImage.transformed(by: CGAffineTransform(scaleX: 128.0 / 750.0, y: 128.0 / 750.0))

            let context = CIContext()
            if let image = context.createCGImage(resizedCIImage, from: resizedCIImage.extent) {
                return UIImage(cgImage: image)
            }
        }
        return nil
    }

    func createBuffer(from image: CIImage) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.extent.width), Int(image.extent.height), kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)

        guard (status == kCVReturnSuccess) else {
            return nil
        }

        return pixelBuffer
    }

    private func processBuffer(buffer: CVPixelBuffer) {

        let input = ModelInput(image: buffer, iouThreshold: 0.01, confidenceThreshold: 0.01)
        do {
            let output = try momomind.prediction(input: input)
            print(output.confidence)
            print(output.coordinates)

        } catch {
            print(error)
        }
    }
}
