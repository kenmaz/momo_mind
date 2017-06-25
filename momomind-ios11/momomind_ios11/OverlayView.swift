//
//  OverlayView.swift
//  momomind_ios11
//
//  Created by Kentaro Matsumae on 2017/06/17.
//  Copyright © 2017年 kenmaz.net. All rights reserved.
//

import UIKit

class OverlayView: UIView {
    
    var boxes:[CGRect]?
    
    override func draw(_ rect: CGRect) {
        let context = UIGraphicsGetCurrentContext()
        
        boxes?.forEach { (box) in
            context?.setStrokeColor(UIColor.red.cgColor)
            context?.setLineWidth(5)
            context?.stroke(box)
        }
    }
}
