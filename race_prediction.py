import cv2
from ultralytics import YOLO
import os
import csv
import numpy as np
import sys
import tensorflow as tf


COUNTRY_PROBABILITIES = None
COUNTRY_PROFILES_DATA = {
    'Ghana':          {'black': 0.98, 'asian': 0.002, 'indian': 0.005, 'others': 0.006, 'white': 0.01},
    'Kenya':          {'black': 0.97, 'asian': 0.003, 'indian': 0.010, 'others': 0.006, 'white': 0.01},
    'South Africa':   {'black': 0.80, 'asian': 0.010, 'indian': 0.030, 'others': 0.030, 'white': 0.10},
    'Japan':          {'black': 0.005, 'asian': 0.98, 'indian': 0.005, 'others': 0.11, 'white': 0.005},
    'China':          {'black': 0.002, 'asian': 0.97, 'indian': 0.005, 'others': 0.10, 'white': 0.005},
    'Iran':           {'black': 0.005, 'asian': 0.01, 'indian': 0.010, 'others': 0.91, 'white': 0.005},
    'Sweden':         {'black': 0.01, 'asian': 0.01, 'indian': 0.020, 'others': 0.05, 'white': 0.90},
    'Czech Republic': {'black': 0.002, 'asian': 0.005, 'indian': 0.002, 'others': 0.02, 'white': 0.95},
    'Austria':        {'black': 0.002, 'asian': 0.005, 'indian': 0.002, 'others': 0.02, 'white': 0.95},
    'United States':  {'black': 0.13, 'asian': 0.06, 'indian': 0.020, 'others': 0.28, 'white': 0.58},
    'Canada':         {'black': 0.03, 'asian': 0.06, 'indian': 0.030, 'others': 0.12, 'white': 0.85},
    'Mexico':         {'black': 0.01, 'asian': 0.005, 'indian': 0.005, 'others': 0.96, 'white': 0.05},
    'Chile':          {'black': 0.01, 'asian': 0.005, 'indian': 0.005, 'others': 0.96, 'white': 0.05},
    'Peru':           {'black': 0.01, 'asian': 0.005, 'indian': 0.005, 'others': 0.96, 'white': 0.05},
    'Argentina':      {'black': 0.01, 'asian': 0.005, 'indian': 0.005, 'others': 0.91, 'white': 0.08},
    'Australia':      {'black': 0.01, 'asian': 0.04, 'indian': 0.030, 'others': 0.04, 'white': 0.85},
    'New Zealand':    {'black': 0.01, 'asian': 0.05, 'indian': 0.050, 'others': 0.05, 'white': 0.80},
    'Fiji':           {'black': 0.002, 'asian': 0.01, 'indian': 0.400, 'others': 0.06, 'white': 0.05},
    'Thailand':       {'black': 0.002, 'asian': 0.95, 'indian': 0.005, 'others': 0.09, 'white': 0.01},
    'France':         {'black': 0.03, 'asian': 0.02, 'indian': 0.020, 'others': 0.06, 'white': 0.80}
}
RACE_NAMES = ['white', 'black', 'asian', 'indian', 'others']


def load_model():
    HUMAN_DETECTION_MODEL = YOLO('fine_tuned_yolov8s.pt')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(script_dir, "deploy.prototxt.txt")
    model_path = os.path.join(script_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    FACE_DETECTION_MODEL = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    if not os.path.exists('race_model.h5'):
        print("downloading model...")
        wget_url = "https://drive.google.com/uc?id=1o-B_kxanT5ynbQgwWtMBhYa6c02nirdt"
        import gdown
        gdown.download(wget_url, 'race_model.h5', quiet=False)
    print("Ładowanie modelu...")
    model_path = os.path.join(script_dir, 'race_model.h5')
    RACE_PREDICTION_MODEL = tf.keras.models.load_model(model_path, compile=False)
    return HUMAN_DETECTION_MODEL, FACE_DETECTION_MODEL, RACE_PREDICTION_MODEL

# ===========================================================
# PERSON DETECTION
# ===========================================================


def detect_and_crop_person(HUMAN_DETECTION_MODEL,image: np.ndarray, conf_threshold: float = 0.1):
    print("Wykrywanie osób...")
    cropped_people = []
    results = HUMAN_DETECTION_MODEL(image, verbose=False)

    for r in results:
        for box in r.boxes:
            if box.conf[0] > conf_threshold:
                class_id = int(box.cls[0])
                class_name = HUMAN_DETECTION_MODEL.names[class_id]
                if class_name == 'pedestrian':
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w, _ = image.shape
                    padding_percent = 0.001
                    padding_x = int((x2 - x1) * padding_percent)
                    padding_y = int((y2 - y1) * padding_percent)

                    x1_pad = max(x1 - padding_x, 0)
                    y1_pad = max(y1 - padding_y, 0)
                    x2_pad = min(x2 + padding_x, w)
                    y2_pad = min(y2 + padding_y, h)
                    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    cropped_people.append(crop)
            else:
                print(f"Osoba wykryta, ale pewność ({box.conf[0]:.2f}) jest poniżej progu ({conf_threshold}).")

    return cropped_people


# ===========================================================
# FACE DETECTION
# ===========================================================

def detect_faces(FACE_DETECTION_MODEL,image, confidence_threshold=0.4):
    print("Wykrywanie jednej twarzy...")
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    FACE_DETECTION_MODEL.setInput(blob)
    detections = FACE_DETECTION_MODEL.forward()

    if detections.shape[2] > 0:
        confidence = detections[0, 0, 0, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            margin_percent = 0.01 
            if margin_percent > 0:
                face_width = endX - startX
                face_height = endY - startY          
                margin_w = int(face_width * margin_percent)
                margin_h = int(face_height * margin_percent)
                startX = startX - (margin_w // 2)
                endX = endX + (margin_w // 2)
                startY = startY - (margin_h // 2)
                endY = endY + (margin_h // 2)
        
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            return image[startY:endY, startX:endX]

    print("Nie wykryto twarzy z wystarczającą pewnością.")
    return None


# ===========================================================
# RACE PREDICTION
# ===========================================================

def predict_race(RACE_PREDICTION_MODEL,image_object):
    country_profiles = COUNTRY_PROFILES_DATA
    
    img_rgb = cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.array(img_resized, dtype='float32')
    processed_image = np.expand_dims(img_array, axis=0)
    
    predictions = RACE_PREDICTION_MODEL.predict(processed_image, verbose=0)
    race_probabilities_vector = predictions[2][0]

    model_output_map = {
        'white': race_probabilities_vector[0],
        'black': race_probabilities_vector[1],
        'asian': race_probabilities_vector[2],
        'indian': race_probabilities_vector[3],
        'others': race_probabilities_vector[4]
    }
    
    country_scores = {}
    for country, demographic_profile in country_profiles.items():
        score = 0.0
        for race_name_from_csv, demographic_prob in demographic_profile.items():
            model_predicted_prob = model_output_map.get(race_name_from_csv, 0.0)
            score += model_predicted_prob * demographic_prob
        country_scores[country] = score

    total_score = sum(country_scores.values())
    if total_score == 0: total_score = 1
    final_probabilities = {country: float(score / total_score) for country, score in country_scores.items()}

    predicted_index = np.argmax(race_probabilities_vector)
    predicted_race_name = RACE_NAMES[predicted_index]
    confidence = race_probabilities_vector[predicted_index]
    label_text = f"{predicted_race_name}"
    
    (h, w, _) = image_object.shape
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w/10.0
    red_color = (0, 0, 255)

    position = (0, 450)

    output_image = image_object.copy()
    output_image = cv2.resize(output_image, (500, 500))
            
    cv2.putText(output_image, 
                label_text, 
                position, 
                font, 
                font_scale, 
                red_color, 
                5, 
                cv2.LINE_AA)
                
    return final_probabilities, output_image



# ===========================================================
# COMBINE PROBABILITIES
# ===========================================================

def combine_probabilities(list_of_probabilities: list[dict]) -> dict:
    print("Łączenie prawdopodobieństw...")

    if not list_of_probabilities:
        # Return zero probabilities for all countries if nothing detected
        return {country: 0.0 for country in COUNTRY_PROFILES_DATA.keys()}

    if len(list_of_probabilities) == 1:
        return list_of_probabilities[0]

    combined_scores = {country: 0.0 for country in list_of_probabilities[0].keys()}

    for prob_dict in list_of_probabilities:
        for country, probability in prob_dict.items():
            if country in combined_scores:
                combined_scores[country] += probability

    total_score = sum(combined_scores.values())
    if total_score == 0:
        return combined_scores

    final_probabilities = {country: score / total_score for country, score in combined_scores.items()}

    return final_probabilities


# ===========================================================
# MAIN FUNCTION 
# ===========================================================



def race_prediction(HUMAN_DETECTION_MODEL, FACE_DETECTION_MODEL,RACE_PREDICTION_MODEL,image):
    person_images = detect_and_crop_person(HUMAN_DETECTION_MODEL, image)
    probabilities = []
    face_images = []
    if person_images:
        for person_image in person_images:
            face_image = detect_faces(FACE_DETECTION_MODEL, person_image)
            if face_image is None:
                print("Nie wykryto twarzy w obrazie.")
                continue
            probability, face_image = predict_race(RACE_PREDICTION_MODEL, face_image)
            probabilities.append(probability)
            face_images.append(face_image)

        return combine_probabilities(probabilities), face_images
    else:
        print("Nie wykryto żadnej osoby na obrazie.")
        return combine_probabilities(probabilities), face_images




# if __name__ == "__main__":
#     probabilities = []
#     face_images = []
#     load_model()

#     image = cv2.imread("image.jpg")
#     probabilities, face_images  = race_prediction(image)

#     print(probabilities)
#     # if probabilities is not None:
#     #     sorted_results = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
#     #     print("--- Posortowane wyniki ---")
#     #     if not sorted_results:
#     #         print("Brak wyników do wyświetlenia.")
#     #     else:
#     #         for kraj, prawdopodobienstwo in sorted_results:
#     #             print(f"{kraj}: {prawdopodobienstwo:.2%}")