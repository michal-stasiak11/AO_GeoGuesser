import cv2
from enum import Enum
import numpy as np
from cv2.typing import MatLike

class Feature(Enum):
        LEFT = 1
        RIGHT = 2
        AMBIGUOUS = 3

class Roadlines(Enum):
    YELLOW_CENTER = 1
    YELLOW_OUTSIDE = 2
    YELLOW_ALL = 3
    YELLOW_NONE = 4
    YELLOW_NONE_CENTER = 5
    YELLOW_NONE_MIXED = 6
    ALL_POSSIBLE = 7
    YELLOW_OUTSIDE_NONE_ALL = 8

selected_countries = {
        "Ghana": (Feature.RIGHT, Roadlines.YELLOW_NONE),
        "Kenya": (Feature.LEFT, Roadlines.YELLOW_CENTER),
        "South Africa": (Feature.LEFT, Roadlines.YELLOW_OUTSIDE),
        "Japan": (Feature.LEFT, Roadlines.YELLOW_NONE_CENTER),
        "China": (Feature.RIGHT, Roadlines.YELLOW_NONE_CENTER),
        "Iran": (Feature.RIGHT, Roadlines.YELLOW_NONE_CENTER),
        "Sweden": (Feature.RIGHT, Roadlines.YELLOW_NONE),
        "Czech Republic": (Feature.RIGHT, Roadlines.YELLOW_NONE),
        "Austria": (Feature.RIGHT, Roadlines.YELLOW_NONE),
        "United States": (Feature.RIGHT, Roadlines.YELLOW_CENTER),
        "Canada": (Feature.RIGHT, Roadlines.YELLOW_CENTER),
        "Mexico": (Feature.RIGHT, Roadlines.YELLOW_NONE_MIXED),
        "Chile": (Feature.RIGHT, Roadlines.ALL_POSSIBLE),
        "Peru": (Feature.RIGHT, Roadlines.YELLOW_NONE_CENTER),
        "Argentina": (Feature.RIGHT, Roadlines.YELLOW_NONE_CENTER),
        "Australia": (Feature.LEFT, Roadlines.YELLOW_OUTSIDE_NONE_ALL),
        "New Zealand": (Feature.LEFT, Roadlines.ALL_POSSIBLE),
        "Fiji": (Feature.LEFT, Roadlines.YELLOW_NONE),
        "Thailand": (Feature.LEFT, Roadlines.YELLOW_CENTER),
        "France": (Feature.RIGHT, Roadlines.YELLOW_NONE)
}

LEFT_COUNTRIES  = set(map(lambda x: x[0], filter(
    lambda x: x[1] == Feature.LEFT,  map(lambda x: (x[0], x[1][0]), selected_countries.items())
)))
RIGHT_COUNTRIES = set(map(lambda x: x[0], filter(
    lambda x: x[1] == Feature.RIGHT, map(lambda x: (x[0], x[1][0]), selected_countries.items())
)))


def plot_bboxes(results) -> MatLike:
    img = results[0].orig_img # original image
    names = results[0].names # class names dict
    scores = results[0].boxes.conf.cpu().numpy() # probabilities
    classes = results[0].boxes.cls.cpu().numpy() # predicted classes
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32) # bboxes
    for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
        class_label = names[cls] # class name
        label = f"{class_label} : {score:0.2f}" # bbox label
        lbl_margin = 3 #label margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),
                            thickness=1)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=0.3, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
                             (bbox[0]+lbl_w, bbox[1]-lbl_h),
                             color=(0, 0, 255), 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3, color=(255, 255, 255 ),
                    thickness=1)
    return img

def weighted_average(d1: dict[str, float], d2: dict[str, float], w1: float, w2: float) -> dict[str, float]:
    d_avg = {c: 0.0 for c in d1.keys()}
    for c in d_avg.keys():
        d_avg[c] = (d1[c] * w1 + d2[c] * w2) / (w1 + w2)
    return d_avg

# Assumes OpenCV image in BGR format
def predict_road_side(image: MatLike, model, conf: float=0.3) -> tuple[dict[str, float], MatLike]:
    """Predicts country probabilities based on the driving side inferred from various signs.
    Model was trained on traffic signs, however with a low `conf` parameter we intentionally 
    allow the detection of similar objects such as ads, signs outside of the dataset etc."""
    h, w, _ = image.shape
    h -= h % 32
    w -= w % 32
    resized_image = cv2.resize(image, dsize=(w, h))

    results = model.predict(source=resized_image, imgsz=(h, w), conf=conf, verbose=False) 

    center_offset = w / 9
    min_size = h * w / 10_000
    left_objs  = 1
    right_objs = 1

    if len(results) != 1:
        raise Exception('Result list should have size 1 for 1 image')
    for result in results:
        for d in result.summary():
            box = d['box']
            box_w, box_h = box['x2'] - box['x1'], box['y2'] - box['y1']
            if box_w * box_h >= min_size:
                if box['x1'] > w // 2 + center_offset:
                    right_objs += 1
                elif box['x2'] < w // 2 - center_offset:
                    left_objs += 1

    result_img = plot_bboxes(results) if results else np.copy(image)
    ratio = max(left_objs, right_objs) / min(left_objs, right_objs)
    preds = {c: 1 / len(list(selected_countries.keys())) for c in selected_countries.keys()}
    if ratio < 1.2:
        pass
    elif left_objs > right_objs:
        preds = weighted_average(preds, {c: 1/len(LEFT_COUNTRIES) if c in LEFT_COUNTRIES else 0.0 for c in selected_countries.keys()}, 1, min(ratio, 3))
    else:
        preds = weighted_average(preds, {c: 1/len(RIGHT_COUNTRIES) if c in RIGHT_COUNTRIES else 0.0 for c in selected_countries.keys()}, 1, min(ratio, 3))

    return (preds, result_img)

