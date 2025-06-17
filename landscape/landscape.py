import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

color_profiles_hsv = {
    "tropical": [
        (140, 100, 80),  # Green
        (200, 100, 80),  # Ocean
    ],
    "desert": [
        (40, 50, 80),    # Sand
        (30, 100, 80),   # Brown
        (120, 25, 60)    # Olive
    ],
    "european": [
        (0, 0, 40),      # Gray
        (90, 50, 60),    # Olive
        (200, 50, 80),   # Sky
    ],
}

def get_dominant_colors(image, k):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v_channel = hsv_image[:, :, 2]
    mask = v_channel > 50
    non_dark_pixels = hsv_image[mask]

    if len(non_dark_pixels) == 0:
        return []

    kmeans = KMeans(n_clusters=k, random_state=0).fit(non_dark_pixels)
    dominant_colors = kmeans.cluster_centers_

    dominant_colors = np.round(dominant_colors).astype(int)

    return dominant_colors

def euclidean_distance(color1, color2):
    return distance.euclidean(color1, color2)

def calculate_category_weights(dominant_colors):
    category_scores = {category: 0 for category in color_profiles_hsv}

    for color in dominant_colors:
        for category, profile_colors in color_profiles_hsv.items():
            distances = [euclidean_distance(color, profile_color) for profile_color in profile_colors]
            category_scores[category] += np.mean(distances)

    squared_scores = {category: score**2 for category, score in category_scores.items()}

    total_score = sum(squared_scores.values())
    category_percentages = {category: (score / total_score) * 100 for category, score in squared_scores.items()}

    return category_percentages

def normalize_dict(input_dict):
    total_sum = sum(input_dict.values())

    normalized_dict = {key: value / total_sum for key, value in input_dict.items()}

    return normalized_dict

def categories_to_countries(percentages):
    countries = {
        "Ghana": 0.5 * percentages['tropical'].item() + 0.4 * percentages['desert'].item() + 0.1 * percentages['european'].item(),
        "Kenya": 0.4 * percentages['tropical'].item() + 0.6 * percentages['desert'].item() + 0.0 * percentages['european'].item(),
        "South Africa": 0.2 * percentages['tropical'].item() + 0.6 * percentages['desert'].item() + 0.2 * percentages['european'].item(),
        "Japan": 0.3 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 0.7 * percentages['european'].item(),
        "China": 0.4 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 0.6 * percentages['european'].item(),
        "Iran": 0.1 * percentages['tropical'].item() + 0.7 * percentages['desert'].item() + 0.2 * percentages['european'].item(),
        "Sweden": 0.0 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 1.0 * percentages['european'].item(),
        "Czech Republic": 0.0 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 1.0 * percentages['european'].item(),
        "Austria": 0.0 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 1.0 * percentages['european'].item(),
        "United States": 0.1 * percentages['tropical'].item() + 0.2 * percentages['desert'].item() + 0.7 * percentages['european'].item(),
        "Canada": 0.0 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 1.0 * percentages['european'].item(),
        "Mexico": 0.2 * percentages['tropical'].item() + 0.7 * percentages['desert'].item() + 0.1 * percentages['european'].item(),
        "Chile": 0.2 * percentages['tropical'].item() + 0.5 * percentages['desert'].item() + 0.3 * percentages['european'].item(),
        "Peru": 0.3 * percentages['tropical'].item() + 0.6 * percentages['desert'].item() + 0.1 * percentages['european'].item(),
        "Argentina": 0.3 * percentages['tropical'].item() + 0.3 * percentages['desert'].item() + 0.4 * percentages['european'].item(),
        "Australia": 0.3 * percentages['tropical'].item() + 0.3 * percentages['desert'].item() + 0.4 * percentages['european'].item(),
        "New Zealand": 0.5 * percentages['tropical'].item() + 0.4 * percentages['desert'].item() + 0.1 * percentages['european'].item(),
        "Fiji": 0.8 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 0.2 * percentages['european'].item(),
        "Thailand": 0.8 * percentages['tropical'].item() + 0.0 * percentages['desert'].item() + 0.2 * percentages['european'].item(),
        "France": 0.1 * percentages['tropical'].item() + 0.1 * percentages['desert'].item() + 0.8 * percentages['european'].item()
    }
    return normalize_dict(countries)

def analyze_image(image):

    dominant_colors = get_dominant_colors(image, k=5)

    category_percentages = calculate_category_weights(dominant_colors)

    return categories_to_countries(category_percentages)