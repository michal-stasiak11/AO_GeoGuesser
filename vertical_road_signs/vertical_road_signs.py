from ultralytics import YOLO
import cv2
import cvzone
import math
import os
import pytesseract
import numpy as np





def detect_shape(cnt, debug_img=None, color_name=None, color_bgr=None):

    area = cv2.contourArea(cnt)
    
    if area < 500: 
        return "too_small", None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    approx_len = len(approx)
    
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int_(box)
    width, height = rect[1]
    
    if height == 0 or width == 0:
        return "invalid", {"area": area}
    
    aspect_ratio = width / height if width > height else height / width
    angle = abs(rect[2])
    
    shape_props = {
        "area": area,
        "vertices": approx_len,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "angle": angle,
        "contour": cnt,
        "approx": approx,
        "box": box
    }
    

    shape_type = "unknown"
    
    if circularity > 0.7 and approx_len >= 6:
        shape_type = "circle"
    elif approx_len == 3:
        shape_type = "triangle"
    elif 0.8 < aspect_ratio < 1.2 and 15 < angle < 75:
        shape_type = "diamond"
    elif approx_len == 4:
        shape_type = "rectangle"
    
    if debug_img is not None and color_bgr is not None:
        if shape_type == "circle":
            cv2.drawContours(debug_img, [cnt], 0, color_bgr, 2)
        elif shape_type == "triangle":
            cv2.drawContours(debug_img, [approx], 0, color_bgr, 2)
        elif shape_type == "diamond":
            cv2.drawContours(debug_img, [box], 0, color_bgr, 2)
        elif shape_type == "rectangle":
            cv2.drawContours(debug_img, [approx], 0, color_bgr, 2)

        margin_x = 5
        margin_y = 20 
        
    
        cv2.putText(debug_img, shape_type.capitalize(), 
                    (margin_x, margin_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
     
            
            
    
    
    return shape_type, debug_img

def analyze_sign_origin(crop_img):
    countries = {
        "Ghana": 0, "Kenya": 0, "South Africa": 0, "Japan": 0, "China": 0, "Iran": 0,
        "Sweden": 0, "Czech Republic": 0, "Austria": 0, "United States": 0, "Canada": 0,
        "Mexico": 0, "Chile": 0, "Peru": 0, "Argentina": 0, "Australia": 0,
        "New Zealand": 0, "Fiji": 0, "Thailand": 0, "France": 0
    }

    region_to_countries = {
        "first_group": ["Sweden", "Czech Republic", "Austria", "France", "China", "South Africa", "Ghana", "Kenya"],
        "second_group": ["United States", "Canada", "Mexico", "Chile", "Peru", "Argentina", "Thailand", "Japan", "Australia", "New Zealand"],
    }



    img = cv2.resize(crop_img, (100, 100))
    debug_img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  

    yellow_mask_opened = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_small)
    blue_mask_opened = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_small)
    red_mask_opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_small)

    yellow_mask_closed = cv2.morphologyEx(yellow_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    blue_mask_closed = cv2.morphologyEx(blue_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    red_mask_closed = cv2.morphologyEx(red_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    yellow_contours, _ = cv2.findContours(yellow_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    images_with_shapes = []

    yellow_circle_count = 0
    yellow_triangle_count = 0
    yellow_diamond_like = False
    yellow_rectangle_count = 0  
    for cnt in yellow_contours:
        shape, debug_img = detect_shape(cnt, debug_img, "yellow", (0, 255, 255))
        if debug_img is not None:
            images_with_shapes.append(debug_img)

        if shape == "circle":
            yellow_circle_count += 1
        elif shape == "triangle":
            yellow_triangle_count += 1
        elif shape == "diamond":
            yellow_diamond_like = True
        elif shape == "rectangle":
            yellow_rectangle_count += 1  

   
    blue_circle_count = 0
    blue_triangle_count = 0
    blue_diamond_like = False
    blue_rectangle_count = 0

    for cnt in blue_contours:
        shape,  debug_img = detect_shape(cnt, debug_img, "blue", (255, 0, 0))
        if debug_img is not None:
            images_with_shapes.append(debug_img)


        if shape == "circle":
            blue_circle_count += 1
        elif shape == "triangle":
            blue_triangle_count += 1
        elif shape == "diamond":
            blue_diamond_like = True
        elif shape == "rectangle":
            blue_rectangle_count += 1

   
    red_circle_count = 0
    red_triangle_count = 0
    red_diamond_like = False
    red_rectangle_count = 0

    for cnt in red_contours:
        shape,  debug_img = detect_shape(cnt, debug_img, "red", (0, 0, 255))
        if debug_img is not None:
            images_with_shapes.append(debug_img)

        if shape == "circle":
            red_circle_count += 1
        elif shape == "triangle":
            red_triangle_count += 1
        elif shape == "diamond":
            red_diamond_like = True
        elif shape == "rectangle":
            red_rectangle_count += 1








    #Do debugowania
    # cv2.namedWindow("Color Detection", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Blue Mask", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Color Detection", 300, 300)
    # cv2.resizeWindow("Yellow Mask", 300, 300)
    # cv2.resizeWindow("Blue Mask", 300, 300)
    # cv2.resizeWindow("Red Mask", 300, 300)
    # cv2.moveWindow("Color Detection", 100, 100)
    # cv2.moveWindow("Yellow Mask", 420, 100)
    # cv2.moveWindow("Blue Mask", 740, 100)
    # cv2.moveWindow("Red Mask", 1060, 100)
    # cv2.imshow("Color Detection", debug_img)
    # cv2.imshow("Yellow Mask", yellow_mask_closed)
    # cv2.imshow("Blue Mask", blue_mask_closed)
    # cv2.imshow("Red Mask", red_mask_closed)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    first_group_score_boost = 0
    second_group_score_boost = 0

    if yellow_diamond_like:
        second_group_score_boost += 10
    if yellow_triangle_count >= 1:
        first_group_score_boost += 2.5
    if yellow_rectangle_count >= 1:
        second_group_score_boost += 10

    if red_circle_count >= 1:
        first_group_score_boost += 2
    if red_triangle_count >= 1:
        first_group_score_boost += 2
    if red_rectangle_count >= 1:
        first_group_score_boost += 2

    if blue_circle_count >= 1:
        first_group_score_boost += 3
    if blue_rectangle_count >= 1:
        first_group_score_boost += 2
        second_group_score_boost += 1   
   

    for country in region_to_countries["first_group"]:
        countries[country] +=  first_group_score_boost
    for country in region_to_countries["second_group"]:
        countries[country] += second_group_score_boost


    return countries, images_with_shapes



def load_model_sign_detection(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    return model

def get_country_prediction_based_on_sign(img, model):

    region_to_countries = {
        "first_group": ["Sweden", "Czech Republic", "Austria", "France", "China", "South Africa", "Ghana", "Kenya"],
        "second_group": ["United States", "Canada", "Mexico", "Chile", "Peru", "Argentina", "Thailand", "Japan", "Australia", "New Zealand"],
    }
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

    class_names = [
        "car", "different traffic sign", "green traffic light", "motorcycle", "pedestrian", "pedestrian crossing",
        "prohibition sign", "red traffic light", "speed limit sign", "truck", "warning sign"
    ]

    road_sign_classes = {
        "different traffic sign",
        "prohibition sign",
        "warning sign",
        "speed limit sign"
    }



    
   
    aggreagated_images_with_shapes = []
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls = class_names[int(box.cls[0])]
            conf = math.floor(box.conf[0] * 100) / 100


            if conf > 0.5 and cls in road_sign_classes:
                h, w, _ = img.shape
                padding_x = int((x2 - x1) * 0.15)
                padding_y = int((y2 - y1) * 0.15)

                x1_pad = max(x1 - padding_x, 0)
                y1_pad = max(y1 - padding_y, 0)
                x2_pad = min(x2 + padding_x, w)
                y2_pad = min(y2 + padding_y, h)

                crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

                countries_local, images_with_shapes = analyze_sign_origin(crop)
                aggreagated_images_with_shapes.extend(images_with_shapes)
        
                for country, score in countries_local.items():
                    if country in countries:
                        countries[country] += score

    first_group_score = countries["Sweden"]
    second_group_score = countries["United States"]
    total = first_group_score + second_group_score
    if total > 0:
        for country in region_to_countries["first_group"]:
            countries[country] = first_group_score/total
        for country in region_to_countries["second_group"]:
            countries[country] = second_group_score/total

    return countries, aggreagated_images_with_shapes







if __name__ == "__main__":

    test_image_path = "zdj.jpg"

    test_image = cv2.imread(test_image_path)
    cv2.imshow("Test Image", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    load_model_path = "fine_tuned_yolov8s.pt"
    model = load_model_sign_detection(load_model_path)
    countries = get_country_prediction_based_on_sign(test_image, model)
    print(countries)

