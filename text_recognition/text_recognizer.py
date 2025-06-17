from langdetect import detect_langs, LangDetectException
import pytesseract
import numpy as np
import cv2
import os
from pathlib import Path

from matplotlib import pyplot as plt

# pretrained data used from: https://github.com/tesseract-ocr/tessdata

textRecognizerPath = Path(os.path.dirname(os.path.realpath(__file__)))
pathToTraineddata = textRecognizerPath / "traineddata"
os.environ['TESSDATA_PREFIX'] = str(pathToTraineddata)

def detectTextRecodnition():
    import platform
    if platform.system() == "Windows":
        tesseractPath = os.getenv("tesseractPath")
        if tesseractPath is None or tesseractPath == '':
            print("TESSERACT IS NOT SUPPORTED ON WINDOWS, DOWNLOAD TESSERACT FOR WINDOWS AND FILL 'tesseractPath' IN .env FILE")
            return False
        
        pytesseract.pytesseract.tesseract_cmd = tesseractPath
        return True
    return True

class TextRecognizer:
    conf_thresh = 0.8
    nms_thresh = 0.4
    inputSize = (320 * 2, 320 * 2)
    textDetectorEAST = cv2.dnn_TextDetectionModel_EAST(pathToTraineddata / "frozen_east_text_detection.pb")

    countriesTrans = {
        "Ghana": 'en',
        "Kenya": 'en',
        "South Africa": "af en",
        "Japan": 'ja',
        "China": 'zh-tw zh-cn',
        "Iran": 'fa',
        "Sweden": 'sv',
        "Czech Republic": "cs sk",
        "Austria": "en",
        "United States": 'en es',
        "Canada": 'en fr',
        "Mexico": 'es',
        "Chile": 'es',
        "Peru": "es",
        "Argentina": "es",
        "Australia": "en",
        "New Zealand": "en",
        "Fiji": "en",
        "Thailand": "th",
        "France": "fr"
    }
    isTesseractLoaded = detectTextRecodnition()

    textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
    textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)


def process(image):
    this = TextRecognizer
    countries = {
        "Ghana": 0,
        "Kenya": 0,
        "South Africa": 0,
        "Japan": 0,
        "China": 0,
        "Iran": 0,
        "Sweden": 0,
        "Czech Republic": 0,
        "Austria": 0,
        "United States": 0,
        "Canada": 0,
        "Mexico": 0,
        "Chile": 0,
        "Peru": 0,
        "Argentina": 0,
        "Australia": 0,
        "New Zealand": 0,
        "Fiji": 0,
        "Thailand": 0,
        "France": 0
    }
    detectedFeatures = []

    if not this.isTesseractLoaded:
        print("TEXT IS NOT ENABLED")
        return countries, detectedFeatures

    image = cv2.resize(image, this.inputSize)

    boxesEAST, textEAST = this.textDetectorEAST.detect(image)

    if len(textEAST) == 0:
        return countries, detectedFeatures

    text = ""

    minVglob, maxVglob = 1000000000, -1000000000
    minHglob, maxHglob = 1000000000, -1000000000
    for box, chances in zip(boxesEAST, textEAST):
        if chances < 0.75:
            continue
            
        minV, maxV = 1000000000, -1000000000
        minH, maxH = 1000000000, -1000000000

        for point in box:
            minV = min(minV, point[1])
            maxV = max(maxV, point[1])
            minH = min(minH, point[0])
            maxH = max(maxH, point[0])
            
            minVglob = min(minVglob, minV)
            maxVglob = max(maxVglob, maxV)
            minHglob = min(minHglob, minH)
            maxHglob = max(maxHglob, maxH)

        minV = max(minV, 0)
        maxV = min(maxV, this.inputSize[1])
        minH = max(minH, 0)
        maxH = min(maxH, this.inputSize[0])

        textImg = image.copy()
        textImg = textImg[minV:maxV, minH:maxH]

        foundText = pytesseract.image_to_string(textImg, lang='eng+jpn+chi_sim+fas+tha+swe+ces+deu+fra+spa+afr').strip()

        text += foundText

    text = text.lower()

    try:
        detection = detect_langs(text)[0]
    except LangDetectException:
        return countries, detectedFeatures
    
    normalizationFactor = 0
    for key, value in this.countriesTrans.items():
        if detection.lang in value:
            countries[key] += 1
            normalizationFactor += 1

    if normalizationFactor != 0:
        for key in countries:
            countries[key] /= normalizationFactor 

    imageWithDetections = image.copy()
    for box in boxesEAST:
        cv2.polylines(imageWithDetections, [np.array(box, np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
    imageWithDetections = imageWithDetections[minVglob:maxVglob, minHglob:maxHglob]
    detectedFeatures.append(imageWithDetections)

    return countries, detectedFeatures


if __name__ == "__main__":
    image1 = cv2.imread("test.png")
    process(image1)
