from dotenv import load_dotenv
load_dotenv()

import random
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import tensorflow as tf
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
import os
import time
from vertical_road_signs.vertical_road_signs import load_model_sign_detection, get_country_prediction_based_on_sign
from landscape.landscape import analyze_image as get_country_prediction_based_on_landscape
from signs_driving_side.signs_driving_side import predict_road_side
from road_lines.road_lines import predict_road_lines
from licenes_plates.license_plate_extractor import detect_license_plates_on_image
from ultralytics import YOLO

from race_prediction import load_model as load_model_race_prediction, race_prediction as get_prediction_based_on_race
from text_recognition import text_recognizer

# Global variables for models and configurations
LOCATION_MODEL = None
LOCATION_CONFIG = None
OBJECT_DETECTION_MODEL = None
TEXT_RECOGNITION_MODEL = None
VERTICAL_ROAD_SIGN_MODEL = None
DRIVING_SIDE_MODEL = None
CAR_DETECTION_MODEL = None
LICENSE_PLATE_MODEL = None
country_decetect_model_from_license_plates = None
config_path =None
license_plate_model =None
HUMAN_DETECTION_MODEL = None
FACE_DETECTION_MODEL = None
RACE_PREDICTION_MODEL = None
# Add more global variables for additional models as needed


def load_models():
    """
    Load all models and configurations at application startup
    
    Returns:
        bool: True if all models loaded successfully, False otherwise
    """
    global LOCATION_MODEL, LOCATION_CONFIG, OBJECT_DETECTION_MODEL, TEXT_RECOGNITION_MODEL, VERTICAL_ROAD_SIGN_MODEL, DRIVING_SIDE_MODEL, CAR_DETECTION_MODEL, LICENSE_PLATE_MODEL, country_decetect_model_from_license_plates, config_path, HUMAN_DETECTION_MODEL, FACE_DETECTION_MODEL, RACE_PREDICTION_MODEL
    
    try:
        load_model_race_prediction()
        print("Loading models...")
        start_time = time.time()
        
        # Check if model files exist
        license_plate_model = 'licenes_plates/license_plate_detector.pt'  # Update with your actual model path
        country_decetect_model_from_license_plates = 'licenes_plates/license_plate_classifier.h5'  
        car_detection_model = 'licenes_plates/yolov8n.pt'
        config_path = 'licenes_plates/model_config.pkl'  # Update with your actual config path
        road_sign_model_path = 'vertical_road_signs/fine_tuned_yolov8s.pt'  # Path to sign detection model
        driving_side_model_path = 'signs_driving_side/30k_20e_yolo11m.pt'

        
        # Optional: Check if files exist before loading
        # if not os.path.exists(model_path) or not os.path.exists(config_path):
        #     print(f"Model files not found. Please ensure models are in the correct location.")
        #     return False
        
        # Load location prediction model (placeholder)
        # In a real application, you would load your actual models here
        print("Loading location prediction model...")
        # LOCATION_MODEL = tf.keras.models.load_model(model_path)
        # Placeholder to simulate model loading time
        time.sleep(1)
        LOCATION_MODEL = "LOCATION_MODEL_PLACEHOLDER"
        
        # Load model configuration (placeholder)
        print("Loading model configuration...")
        # with open(config_path, 'rb') as f:
        #     LOCATION_CONFIG = pickle.load(f)
        # Placeholder to simulate config loading time
        time.sleep(0.5)

        
        # Load object detection model (placeholder)
        print("Loading object detection model...")
        # OBJECT_DETECTION_MODEL = YourObjectDetectionFramework.load("models/detection_model.pt")
        time.sleep(1)
        OBJECT_DETECTION_MODEL = "OBJECT_DETECTION_MODEL_PLACEHOLDER"
        
        print("Loading text recognition model...")
        
        print("Loading Race prediction models detection model...")
        try:
            HUMAN_DETECTION_MODEL, FACE_DETECTION_MODEL, RACE_PREDICTION_MODEL = load_model_race_prediction()
            print("Race prediction model loaded successfully")
        except Exception as e:
            print(f"Error loading Race prediction model: {e}")
            print("You need to download model manually from https://drive.google.com/file/d/1o-B_kxanT5ynbQgwWtMBhYa6c02nirdt/view?usp=sharing")
            HUMAN_DETECTION_MODEL = None
            FACE_DETECTION_MODEL = None
            RACE_PREDICTION_MODEL = None

        # Add more model loading code here as needed
        print("Loading road sign detection model...")
        try:
            if os.path.exists(road_sign_model_path):
                VERTICAL_ROAD_SIGN_MODEL = load_model_sign_detection(road_sign_model_path)
                print("Road sign model loaded successfully")
            else:
                print(f"Road sign model file not found at {road_sign_model_path}")
                # Create placeholder model for testing purposes
                VERTICAL_ROAD_SIGN_MODEL = "ROAD_SIGN_MODEL_PLACEHOLDER"
        except Exception as e:
            print(f"Error loading road sign model: {e}")
            VERTICAL_ROAD_SIGN_MODEL = None

        # Driving side detection model
        print("Loading driving side detection model...")
        try:
            if os.path.exists(driving_side_model_path):
                DRIVING_SIDE_MODEL = YOLO(driving_side_model_path)
                print("Driving side model loaded_successfully")
            else:
                print(f"Driving side model file not found at: {driving_side_model_path}")
        except Exception as e:
            print(f'Error loading driving side model: {e}')
            DRIVING_SIDE_MODEL = None
        
        print("Loading license plate detection model...")
        try:
            if os.path.exists(car_detection_model):
                CAR_DETECTION_MODEL = YOLO(car_detection_model)
                print("Car detection model loaded successfully")
            else:
                print(f"Car detection model file not found at: {car_detection_model}")
            if os.path.exists(license_plate_model):
                LICENSE_PLATE_MODEL = YOLO(license_plate_model)
                print("License plate model loaded successfully")
            else:
                print(f"License plate model file not found at: {license_plate_model}")
        except Exception as e:
            print(f"Error loading license plate model: {e}")
            CAR_DETECTION_MODEL = None
            LICENSE_PLATE_MODEL = None
        
        elapsed_time = time.time() - start_time
        print(f"All models loaded successfully in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return 
        
# Define weights for each prediction model
weights = {
    "landscape": 0.03,   
    "road_signs": 0.30,   
    "driving_side": 0.06, 
    "road_lines": 0.11,   
    "race": 0.15,         
    "license_plates" : 0.20,
    "recognized_text": 0.20,
}

# Function to normalize country scores
def normalize_scores(country_dict):
    total = sum(country_dict.values())
    if total > 0:
        return {country: score/total for country, score in country_dict.items()}
    return country_dict

def predict_location(image, model_path=None):
    global LOCATION_MODEL, LOCATION_CONFIG, VERTICAL_ROAD_SIGN_MODEL, LICENSE_PLATE_MODEL, CAR_DETECTION_MODEL, config_path, country_decetect_model_from_license_plates
    
    # Use the fixed set of countries
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
    
    # Convert PIL image to OpenCV format for road sign detection
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detected_objects = {}
    
    # Initialize categories
    categories = ["Humans", "Vertical Road Signs", "License Plates", "Driving side", "Road lines", "Recognized Text"]
    for category in categories:
        detected_objects[category] = []
    
    # ---- Text recognition ----
    countries_T, features_from_text = text_recognizer.process(img_cv)
    detected_objects["Recognized Text"].append(features_from_text)
    # ----------------------

    countries_VRS, detected_objects_VRS = get_country_prediction_based_on_sign(img_cv, VERTICAL_ROAD_SIGN_MODEL)

    detected_objects["Vertical Road Signs"] = detected_objects_VRS

    # ---- Driving side ----
    if DRIVING_SIDE_MODEL is not None:
        countries_RS, image_RS = predict_road_side(img_cv, DRIVING_SIDE_MODEL)
        detected_objects["Driving side"].append(image_RS)
    # ----------------------

    # ---- Road lines  ----
    countries_RL, image_RL = predict_road_lines(img_cv)
    detected_objects["Road lines"].append(image_RL)
    # ---------------------

    # ---- License plates ----
    countries_LP, image_LP = detect_license_plates_on_image(img_cv, LICENSE_PLATE_MODEL, CAR_DETECTION_MODEL, config_path, country_decetect_model_from_license_plates)
    detected_objects["License Plates"].extend(image_LP)
    if not countries_LP:
        countries_LP = countries
    # ------------------------

    countries_L = get_country_prediction_based_on_landscape(img_cv)

    countries_R, image_R = get_prediction_based_on_race( HUMAN_DETECTION_MODEL, FACE_DETECTION_MODEL, RACE_PREDICTION_MODEL, img_cv)
    detected_objects["Humans"].extend(image_R)
    print(len(image_R))
    
    # Initialize countries_final with zeros for all countries
    countries_final = {country: 0 for country in countries_L.keys()}

    # Add weighted landscape predictions
    normalized_L = normalize_scores(countries_L)
    for country, score in normalized_L.items():
        countries_final[country] += score * weights["landscape"]

    # Add weighted road sign predictions if available
    if countries_VRS:
        normalized_VRS = normalize_scores(countries_VRS)
        for country, score in normalized_VRS.items():
            if country in countries_final:  # Ensure country exists in final dict
                countries_final[country] += score * weights["road_signs"]

    # Add weighted driving side predictions if available
    if DRIVING_SIDE_MODEL is not None and countries_RS:
        normalized_RS = normalize_scores(countries_RS)
        for country, score in normalized_RS.items():
            if country in countries_final:
                countries_final[country] += score * weights["driving_side"]

    # Add weighted road lines predictions if available
    if countries_RL:
        normalized_RL = normalize_scores(countries_RL)
        for country, score in normalized_RL.items():
            if country in countries_final:
                countries_final[country] += score * weights["road_lines"]

    if countries_R:
        normalized_R = normalize_scores(countries_R)
        for country, score in normalized_R.items():
            if country in countries_final:
                countries_final[country] += score * weights["race"]

    if countries_LP:
        normalized_LP = normalize_scores(countries_LP)
        for country, score in normalized_LP.items():
            if country in countries_final:
                countries_final[country] += score * weights["license_plates"]

    if countries_T:
        normalized_T = normalize_scores(countries_T)
        for country, score in normalized_T.items():
            if country in countries_final:
                countries_final[country] += score * weights["recognized_text"]

    # Normalize final scores to ensure they sum to 1
    countries_final = normalize_scores(countries_final)

    # Optional: Print combined scores
    print("Final weighted country predictions:")
    for country, score in sorted(countries_final.items(), key=lambda x: x[1], reverse=True):
        print(f"{country}: {score:.4f}")

    return countries_final, detected_objects


def predict_batch(image_list, model_path=None):
    if not image_list:
        return {}, {}
        
    # Make predictions for each image
    all_predictions = []
    all_detected_objects = []
    
    for img in image_list:
        location_preds, detected_objs = predict_location(img, model_path)
        all_predictions.append(location_preds)
        all_detected_objects.append(detected_objs)
    
    # Aggregate location predictions (average)
    countries = all_predictions[0].keys()
    aggregated_locations = {country: 0.0 for country in countries}
    
    for pred in all_predictions:
        for country, prob in pred.items():
            aggregated_locations[country] += prob
    
    # Normalize
    for country in aggregated_locations:
        aggregated_locations[country] /= len(all_predictions)
    
    # Aggregate object detections (combine all detections)
    aggregated_objects = {}
    
    # Get all unique categories across all frames
    all_categories = set()
    for objs in all_detected_objects:
        all_categories.update(objs.keys())
    
    # Combine all detections for each category
    for category in all_categories:
        aggregated_objects[category] = []
        for objs in all_detected_objects:
            if category in objs:
                aggregated_objects[category].extend(objs[category])
                
    return aggregated_locations, aggregated_objects


class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analyzer Template")
        self.root.state('normal')
        
        # Stałe rozmiary
        self.VIDEO_WIDTH = 800
        self.VIDEO_HEIGHT = 450
        
        # Zmienne
        self.video_path = ""
        self.cap = None
        self.is_playing = False
        self.analysis_mode = False
        self.current_frame = None  
        
        self.video_duration = 0  # Total duration in seconds
        self.video_fps = 0       # Video FPS
        self.total_frames = 0    # Total frames count
        
        # Variables for analysis range
        self.start_second = tk.DoubleVar(value=0)
        self.end_second = tk.DoubleVar(value=0)
        self.frame_sample_rate = tk.IntVar(value=1)  
        
        self.detection_categories = [
            "Humans",
            "Vertical Road Signs",
            "License Plates",
            "Driving side",
            "Road lines",
            "Recognized Text"
        ]
        
        self.detected_data = {
            category: [] for category in self.detection_categories
        }
        self.current_indices = {key: 0 for key in self.detected_data.keys()}
        
        # Check if models are loaded before starting
        self.models_loaded = LOCATION_MODEL is not None
        
        # GUI elements
        self.create_widgets()
        
    def generate_sample_image(self):
        """Generate sample image for testing purposes"""
        img = Image.new('RGB', (150, 100), color=(random.randint(100, 200), 
                                              random.randint(100, 200), 
                                              random.randint(100, 200)))
        return ImageTk.PhotoImage(img)
        
    def create_widgets(self):
        # Główny kontener
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model status indicator
        self.model_status_frame = tk.Frame(self.main_frame)
        self.model_status_frame.pack(fill=tk.X, pady=2)
        
        model_status_text = "Models: " + ("Loaded ✓" if self.models_loaded else "Not loaded ✗")
        model_status_color = "green" if self.models_loaded else "red"
        
        self.lbl_model_status = tk.Label(self.model_status_frame, text=model_status_text, 
                                        fg=model_status_color, font=('Arial', 9))
        self.lbl_model_status.pack(side=tk.RIGHT, padx=10)
        
        # Górna część (wideo + ranking)
        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results panel (right side)
        self.results_frame = tk.Frame(self.top_frame, width=500, bg='#f0f0f0')
        self.results_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.results_frame.pack_propagate(False)  # Prevent the frame from shrinking
        
        # Results header
        self.lbl_results = tk.Label(self.results_frame, text="Analysis Results", 
                                    font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.lbl_results.pack(pady=10)
        
        # Results list
        self.results_list = tk.Listbox(self.results_frame, width=30, height=20, font=('Arial', 11))
        self.results_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.results_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_list.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_list.yview)
        
        # Panel wideo (lewa strona)
        self.video_frame = tk.Frame(self.top_frame, bg='white')
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Kontener dla przycisków
        self.button_frame = tk.Frame(self.video_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # Przyciski wideo
        self.btn_open = tk.Button(self.button_frame, text="Load Video", command=self.open_video)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = tk.Button(self.button_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        # Button to analyze current frame
        self.btn_analyze_frame = tk.Button(self.button_frame, text="Analyze Frame", 
                                            command=self.analyze_current_frame, state=tk.DISABLED)
        self.btn_analyze_frame.pack(side=tk.LEFT, padx=5)
        
        # Button to analyze video range
        self.btn_analyze = tk.Button(self.button_frame, text="Analyze Video", command=self.toggle_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)
        
        # Button to configure analysis range
        self.btn_range_config = tk.Button(self.button_frame, text="Analysis Range", command=self.show_range_dialog, state=tk.DISABLED)
        self.btn_range_config.pack(side=tk.LEFT, padx=5)
        
        # Video display label
        self.lbl_video = tk.Label(self.video_frame, bg='black')
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom section (detection boxes)
        self.bottom_frame = tk.Frame(self.main_frame, height=250, bg='#e0e0e0')
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Display initial status in results list
        if self.models_loaded:
            self.results_list.insert(tk.END, "Models loaded successfully")
            self.results_list.insert(tk.END, "Ready for analysis")
            self.results_list.insert(tk.END, "Please load a video file to begin")
        else:
            self.results_list.insert(tk.END, "WARNING: Models not loaded")
            self.results_list.insert(tk.END, "Some functionality may be limited")
            self.results_list.insert(tk.END, "Please check console for errors")

    def create_detection_boxes(self):
        """Create UI boxes for displaying detected objects"""
        # Ensure the bottom frame is clear
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()
            
        box_titles = {
            "road_signs": "Road Signs",
            "license_plates": "License Plates",
            "buildings": "Buildings",
            "landmarks": "Landmarks", 
            "text": "RecognizedText"
        }
        
        self.detection_boxes = []
        for i, category in enumerate(self.detection_categories):
            title = box_titles.get(category, f"Detected {category}")
            
            box_frame = tk.Frame(self.bottom_frame, width=400, height=350, 
                                bd=2, relief=tk.GROOVE, bg='white')
            box_frame.pack_propagate(False)
            box_frame.grid(row=0, column=i, padx=5, pady=5, sticky='nsew')
            
            # Box title
            lbl_title = tk.Label(box_frame, text=title, font=('Arial', 10, 'bold'), bg='white')
            lbl_title.pack(pady=5)
            
            # Create an image container with fixed height
            img_container = tk.Frame(box_frame, bg='white', height=150)
            img_container.pack(fill=tk.X, pady=5)
            img_container.pack_propagate(False)  # Prevent resizing
            
            # Image label inside the container
            img_label = tk.Label(img_container, bg='white')
            img_label.pack(expand=True)
            
            # Navigation buttons
            btn_frame = tk.Frame(box_frame, bg='white', height=30)
            btn_frame.pack(fill=tk.X)
            
            btn_prev = tk.Button(btn_frame, text="Previous", width=10,
                               command=lambda k=category: self.prev_image(k))
            btn_prev.pack(side=tk.LEFT, padx=10)
            
            btn_next = tk.Button(btn_frame, text="Next", width=10,
                               command=lambda k=category: self.next_image(k))
            btn_next.pack(side=tk.RIGHT, padx=10)
            
            # Info label
            info_label = tk.Label(box_frame, text="", font=('Arial', 9), bg='white')
            info_label.pack(pady=(5, 5))

            # Store references to UI elements
            self.detection_boxes.append({
                'category': category,
                'frame': box_frame,
                'label': img_label,
                'prev_btn': btn_prev,
                'next_btn': btn_next,
                'info_label': info_label
            })
            
            self.bottom_frame.grid_columnconfigure(i, weight=1)
        
        # Initialize with first images
        self.update_detection_boxes()
    
    def update_detection_boxes(self):
        for box in self.detection_boxes:
            category = box['category']
            
            # If there are detected objects in this category
            if self.detected_data[category]:
                current_idx = self.current_indices[category]
                img = self.detected_data[category][current_idx]
                
                # Check if the image is already a PhotoImage or needs conversion
                if isinstance(img, ImageTk.PhotoImage):
                    # It's already a PhotoImage, use directly
                    box['label'].config(image=img)
                    box['label'].image = img  # Keep a reference to avoid garbage collection
                elif isinstance(img, np.ndarray):
                    # It's an OpenCV image (numpy array), convert to PhotoImage
                    try:
                        # Convert from BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Convert to PIL Image
                        pil_img = Image.fromarray(img_rgb)
                        # Resize if needed
                        pil_img.thumbnail((150, 100))
                        # Convert to PhotoImage
                        photo_img = ImageTk.PhotoImage(pil_img)
                        # Display and keep reference
                        box['label'].config(image=photo_img)
                        box['label'].image = photo_img
                        # Update the stored image to avoid converting next time
                        self.detected_data[category][current_idx] = photo_img
                    except Exception as e:
                        print(f"Error converting image for {category}: {e}")
                        box['label'].config(image="")
                else:
                    # Unknown format, clear the image
                    box['label'].config(image="")
                
                box['prev_btn'].config(state=tk.NORMAL if current_idx > 0 else tk.DISABLED)
                box['next_btn'].config(state=tk.NORMAL if current_idx < len(self.detected_data[category])-1 else tk.DISABLED)
                box['info_label'].config(text=f"Object {current_idx+1} of {len(self.detected_data[category])}")
            else:
                # No objects in this category
                box['label'].config(image="")
                box['prev_btn'].config(state=tk.DISABLED)
                box['next_btn'].config(state=tk.DISABLED)
                box['info_label'].config(text="No objects detected")
        
    def prev_image(self, category):
        """Show previous image in the specified category"""
        if self.current_indices[category] > 0:
            self.current_indices[category] -= 1
            self.update_detection_boxes()
    
    def next_image(self, category):
        """Show next image in the specified category"""
        if self.current_indices[category] < len(self.detected_data[category])-1:
            self.current_indices[category] += 1
            self.update_detection_boxes()
    
    def show_range_dialog(self):
        """Show dialog for setting analysis range"""
        if self.cap is None:
            messagebox.showerror("Error", "Please load a video file first")
            return
            
        # Create a toplevel window
        dialog = tk.Toplevel(self.root)
        dialog.title("Analysis Range")
        dialog.geometry("400x250")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame for content
        content_frame = tk.Frame(dialog, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video info
        video_info_text = f"Video duration: {self.video_duration:.2f}s, FPS: {self.video_fps:.2f}"
        lbl_video_info = tk.Label(content_frame, text=video_info_text, font=('Arial', 10))
        lbl_video_info.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 15))
        
        # Start time
        lbl_start = tk.Label(content_frame, text="Start time (s):")
        lbl_start.grid(row=1, column=0, sticky='w', pady=5)
        
        start_entry = ttk.Spinbox(
            content_frame, 
            from_=0, 
            to=self.video_duration, 
            increment=0.5,
            textvariable=self.start_second,
            width=10
        )
        start_entry.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # End time
        lbl_end = tk.Label(content_frame, text="End time (s):")
        lbl_end.grid(row=2, column=0, sticky='w', pady=5)
        
        end_entry = ttk.Spinbox(
            content_frame, 
            from_=0, 
            to=self.video_duration, 
            increment=0.5,
            textvariable=self.end_second,
            width=10
        )
        end_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Sample rate
        lbl_rate = tk.Label(content_frame, text="Frames per second:")
        lbl_rate.grid(row=3, column=0, sticky='w', pady=5)
        
        rate_entry = ttk.Spinbox(
            content_frame, 
            from_=1, 
            to=self.video_fps,
            increment=1,
            textvariable=self.frame_sample_rate,
            width=10
        )
        rate_entry.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # Buttons
        btn_cancel = tk.Button(content_frame, text="Cancel", width=10, command=dialog.destroy)
        btn_cancel.grid(row=4, column=0, sticky='w', pady=20)
        
        btn_ok = tk.Button(content_frame, text="OK", width=10, 
                          command=lambda: self.validate_and_close_dialog(dialog))
        btn_ok.grid(row=4, column=1, padx=5, pady=20)
        
        # Set end time to video duration if it's zero (initial state)
        if self.end_second.get() == 0:
            self.end_second.set(self.video_duration)
    
    def validate_and_close_dialog(self, dialog):
        """Validate range settings and close dialog"""
        start = self.start_second.get()
        end = self.end_second.get()
        rate = self.frame_sample_rate.get()
        
        if start >= end:
            messagebox.showerror("Error", "Start time must be less than end time")
            return
            
        if start < 0 or end > self.video_duration:
            messagebox.showerror("Error", f"Time range must be between 0-{self.video_duration}s")
            return
            
        if rate < 1 or rate > self.video_fps:
            messagebox.showerror("Error", f"Frame rate must be between 1-{int(self.video_fps)}")
            return
            
        dialog.destroy()
        
    def open_video(self):
        """Open video file and initialize video player"""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if self.video_path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            
            # Get video information
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
            
            # Set default analysis range
            self.start_second.set(0)
            self.end_second.set(self.video_duration)
            self.frame_sample_rate.set(min(1, int(self.video_fps)))
            
            # Enable buttons
            self.btn_analyze.config(state=tk.NORMAL)
            self.btn_play.config(state=tk.NORMAL)
            self.btn_analyze_frame.config(state=tk.NORMAL)
            self.btn_range_config.config(state=tk.NORMAL)
            
            # Show video info in results
            self.results_list.delete(0, tk.END)
            self.results_list.insert(tk.END, f"Loaded video: {os.path.basename(self.video_path)}")
            self.results_list.insert(tk.END, f"Duration: {self.video_duration:.2f} seconds")
            self.results_list.insert(tk.END, f"FPS: {self.video_fps:.2f}")
            self.results_list.insert(tk.END, f"Total frames: {self.total_frames}")
            
            self.show_frame()
    
    def show_frame(self):
        """Display current video frame"""
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                # Store current frame for analysis
                self.current_frame = frame.copy()
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                video_width = self.video_frame.winfo_width() - 10
                video_height = self.video_frame.winfo_height() - 50  
                
                img.thumbnail((video_width, video_height))
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.lbl_video.config(image=imgtk)
                self.lbl_video.image = imgtk
                
                self.lbl_video.after(30, self.show_frame)
            else:
                # Video ended
                self.is_playing = False
                self.btn_play.config(text="Play")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif self.cap and not self.is_playing:
            # Show single frame when paused
            if self.current_frame is not None:
                frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                video_width = self.video_frame.winfo_width() - 10
                video_height = self.video_frame.winfo_height() - 50  
                
                img.thumbnail((video_width, video_height))
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.lbl_video.config(image=imgtk)
                self.lbl_video.image = imgtk
    
    def toggle_play(self):
        """Toggle video playback"""
        if self.cap is None:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="Pause")
            self.show_frame()
        else:
            self.btn_play.config(text="Play")
    
    def analyze_current_frame(self):
        """Analyze current frame that is displayed"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame to analyze")
            return
            
        # Pause video if playing
        if self.is_playing:
            self.toggle_play()
            
        # Show analysis in progress
        self.btn_analyze_frame.config(text="Analyzing...", state=tk.DISABLED)
        self.root.update()
        
        try:
            # Convert current frame to PIL Image for analysis
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Use the prediction function to analyze the frame
            location_predictions, detected_objects = predict_location(frame_pil)
            
            # Update detection data with returned objects
            self.detected_data = detected_objects
            self.current_indices = {category: 0 for category in self.detected_data.keys()}
            
            # Create or update the detection boxes
            self.create_detection_boxes()
            
            # Update results list with prediction results
            self.update_results_list_with_predictions(location_predictions, "Single frame analysis")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Restore button state
        self.btn_analyze_frame.config(text="Analyze Frame", state=tk.NORMAL)
    
    def toggle_analysis(self):
        """Analyze video in the specified range"""
        if not self.video_path:
            messagebox.showerror("Error", "Please load a video file first")
            return
            
        self.analysis_mode = not self.analysis_mode
        if self.analysis_mode:
            self.btn_analyze.config(text="Analyzing...", state=tk.DISABLED)
            self.root.update()
            
            start_frame = int(self.start_second.get() * self.video_fps)
            end_frame = int(self.end_second.get() * self.video_fps)
            sample_rate = self.frame_sample_rate.get()
            frame_step = max(1, int(self.video_fps / sample_rate))
            
            try:
                # Extract frames from the video in the specified range
                frames_pil = self.extract_frames_from_video(start_frame, end_frame, frame_step)
                
                if frames_pil:
                    # Use the batch prediction function to analyze the frames
                    location_predictions, detected_objects = predict_batch(frames_pil)
                    
                    # Update detection data with returned objects
                    self.detected_data = detected_objects
                    self.current_indices = {category: 0 for category in self.detected_data.keys()}
                    
                    # Create or update detection boxes
                    self.create_detection_boxes()
                    
                    # Update results list
                    analysis_summary = f"Video analysis from {self.start_second.get():.1f}s to {self.end_second.get():.1f}s, {sample_rate} fps"
                    self.update_results_list_with_predictions(location_predictions, analysis_summary)
                else:
                    messagebox.showinfo("Info", "No frames were extracted for analysis")
                
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                
            self.analysis_mode = False
            self.btn_analyze.config(text="Analyze Video", state=tk.NORMAL)
    
    def extract_frames_from_video(self, start_frame, end_frame, frame_step):
        """Extract frames from video for analysis"""
        frames_pil = []
        
        if self.cap is None:
            return frames_pil
            
        # Store current position to restore later
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Set position to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frame_count = start_frame
        while frame_count <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process only frames at step intervals
            if (frame_count - start_frame) % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames_pil.append(pil_img)
                
            frame_count += 1
            
        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        return frames_pil
    
    def update_results_list_with_predictions(self, predictions, summary_text):
        """Update results list with predictions"""
        self.results_list.delete(0, tk.END)
        
        # Add summary
        self.results_list.insert(tk.END, summary_text)
        self.results_list.insert(tk.END, "-" * 30)
        
        # Add country predictions, sorted by probability
        self.results_list.insert(tk.END, "Location Predictions:")
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        for country, prob in sorted_preds:
            # Format probability as percentage with 1 decimal
            prob_percent = f"{prob*100:.1f}%"
            self.results_list.insert(tk.END, f"{country}: {prob_percent}")
            

        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results_list.insert(tk.END, "-" * 30)
        self.results_list.insert(tk.END, f"Analysis time: {timestamp}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    # Load models before starting the application
    models_loaded = load_models()
    
    # Start the application
    root = tk.Tk()
    
    try:
        app = VideoAnalyzerApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        root.destroy()
