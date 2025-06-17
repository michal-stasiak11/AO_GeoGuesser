import cv2
import numpy as np
import tensorflow as tf
import pickle

def predict_country_batch(image_list_cv2, model_path='license_plate_classifier.h5', config_path='model_config.pkl'):
    model = tf.keras.models.load_model(model_path)
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    classes = config['classes']
    img_height = config['img_height']
    img_width = config['img_width']

    processed_images = []

    for img_bgr in image_list_cv2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_width, img_height))
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)

    batch_array = np.array(processed_images)

    if batch_array.shape[0] == 0:
        return {}
    probs_batch = model.predict(batch_array)
    avg_probs = np.mean(probs_batch, axis=0)

    return {str(classes[i]): float(avg_probs[i]) for i in range(len(classes))}


def detect_license_plates_on_image(cv2_image, license_plate_model, coco_model, config, model_country_detect):
    vehicle_class_ids = [2, 3, 5, 7]

    detected_plate_images = []

    detections = coco_model(cv2_image)[0]
    vehicle_boxes = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicle_class_ids:
            vehicle_boxes.append([x1, y1, x2, y2])

    license_plates = license_plate_model(cv2_image)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        plate_crop = cv2_image[int(y1):int(y2), int(x1):int(x2)]

        detected_plate_images.append(plate_crop)
    
    detected_plates_countries = predict_country_batch(detected_plate_images, model_country_detect, config)
    print(len(detected_plate_images), "\n\ndetected plates")

    return detected_plates_countries, detected_plate_images
