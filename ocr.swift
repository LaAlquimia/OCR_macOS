import Foundation
import Vision
import AppKit

func performOCR(imagePath: String) {
    let fileURL = URL(fileURLWithPath: imagePath)
    guard let image = NSImage(contentsOf: fileURL),
          let tiffData = image.tiffRepresentation,
          let ciImage = CIImage(data: tiffData) else {
        print("Error: No se pudo cargar la imagen desde \(imagePath)")
        exit(1)
    }

    let requestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    
    let request = VNRecognizeTextRequest { request, error in
        if let error = error {
            print("Error en reconocimiento: \(error.localizedDescription)")
            return
        }
        
        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            return
        }
        
        var recognizedText = ""
        for observation in observations {
            guard let topCandidate = observation.topCandidates(1).first else { continue }
            recognizedText += topCandidate.string + "\n"
        }
        
        print(recognizedText)
    }
    
    // Configurar para usar máxima calidad (reconocimiento preciso) y soporte de idiomas
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    
    do {
        try requestHandler.perform([request])
    } catch {
        print("Error al ejecutar la solicitud: \(error.localizedDescription)")
    }
}

let arguments = CommandLine.arguments
if arguments.count < 2 {
    print("Uso: ocr <ruta_de_imagen>")
    exit(1)
}

let imagePath = arguments[1]
performOCR(imagePath: imagePath)
