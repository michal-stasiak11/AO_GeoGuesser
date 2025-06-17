import cv2
import numpy as np
import math
from cv2.typing import MatLike
from typing import Sequence
from signs_driving_side.signs_driving_side import Feature, Roadlines, selected_countries

def country_profile(country: str) -> dict[str, float]:
    if country not in selected_countries.keys():
        raise Exception(f'Uknown country: {country}')
        #return {c: 1/len(list(selected_countries.keys())) for c in selected_countries.keys()}
    side, line_feature = selected_countries[country]
    if (side, line_feature) in [(Feature.LEFT, Roadlines.YELLOW_OUTSIDE), (Feature.RIGHT, Roadlines.YELLOW_CENTER)]:
        l_yellow_ratio = 0.45
        r_yellow_ratio = l_white_ratio = 0.05
        r_white_ratio = 0.45
        lvr_yellow = 0.4
        lvr_white = -0.4
    elif (side, line_feature) in [(Feature.LEFT, Roadlines.YELLOW_CENTER), (Feature.RIGHT, Roadlines.YELLOW_OUTSIDE)]:
        r_yellow_ratio = 0.45
        l_yellow_ratio = r_white_ratio = 0.05
        l_white_ratio = 0.45
        lvr_yellow = -0.4
        lvr_white = 0.4
    elif line_feature == Roadlines.YELLOW_ALL:
        r_yellow_ratio = l_yellow_ratio = 0.5
        l_white_ratio = r_white_ratio = 0.0
        lvr_yellow = 0.0 
        lvr_white = 0.0
    elif (side, line_feature) in [(Feature.LEFT, Roadlines.YELLOW_NONE), (Feature.RIGHT, Roadlines.YELLOW_NONE)]:
        r_yellow_ratio = l_yellow_ratio = 0.0
        l_white_ratio = r_white_ratio = 0.5
        lvr_yellow = 0.0 
        lvr_white = 0.0
    elif (side, line_feature) == (Feature.LEFT, Roadlines.YELLOW_NONE_CENTER):
        l_yellow_ratio = 0.01
        r_yellow_ratio = 0.15
        l_white_ratio = 0.69
        r_white_ratio = 0.15
        lvr_yellow = -0.4
        lvr_white = 0.1
    elif (side, line_feature) == (Feature.RIGHT, Roadlines.YELLOW_NONE_CENTER):
        l_yellow_ratio = 0.15
        r_yellow_ratio = 0.01
        l_white_ratio = 0.15
        r_white_ratio = 0.69
        lvr_yellow = 0.4
        lvr_white = -0.1
    elif line_feature == Roadlines.YELLOW_NONE_MIXED:
        l_yellow_ratio = 0.15
        r_yellow_ratio = 0.15
        l_white_ratio = 0.35
        r_white_ratio = 0.35
        lvr_yellow = 0.0
        lvr_white = 0.0
    elif line_feature == Roadlines.ALL_POSSIBLE:
        l_yellow_ratio = 0.25
        r_yellow_ratio = 0.25
        l_white_ratio = 0.25
        r_white_ratio = 0.25
        lvr_yellow = 0.0
        lvr_white = 0.0
    elif (side, line_feature) == (Feature.LEFT, Roadlines.YELLOW_OUTSIDE_NONE_ALL):
        l_yellow_ratio = 0.30
        r_yellow_ratio = 0.15
        l_white_ratio = 0.15
        r_white_ratio = 0.40
        lvr_yellow = 0.3
        lvr_white = -0.3
    elif (side, line_feature) == (Feature.RIGHT, Roadlines.YELLOW_OUTSIDE_NONE_ALL):
        l_yellow_ratio = 0.15
        r_yellow_ratio = 0.30
        l_white_ratio = 0.40
        r_white_ratio = 0.15
        lvr_yellow = -0.3
        lvr_white = 0.3
    else:
        raise Exception(f'Uknown (side, line_feature): {(side, line_feature)}')

    return {
        'left_yellow_ratio': l_yellow_ratio,
        'left_white_ratio': l_white_ratio,
        'right_yellow_ratio': r_yellow_ratio,
        'right_white_ratio': r_white_ratio,
        'left_vs_right_yellow': lvr_yellow,
        'left_vs_right_white': lvr_white
    }

def get_country_profiles() -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for country in selected_countries.keys():
        profiles[country] = country_profile(country)
    return profiles

COUNTRY_PROFILES = get_country_profiles()

def grayscale(img: MatLike):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def convert_hls(img: MatLike):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def canny(img: MatLike, low_threshold: int, high_threshold: int):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img: MatLike, kernel_size: int):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img: MatLike, vertices: Sequence[MatLike]):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        mask_color = (255,) * channel_count
        print(mask_color)

    else:
        mask_color = (255,)

    _ = cv2.fillPoly(mask, vertices, mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img: MatLike, lines: MatLike, color: list[int]=[255, 0, 0], thickness: int=2):
    if lines is None:
        return
    for line in lines:
        for points in line:
            x1, y1, x2, y2 = points
            _ = cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img: MatLike, initial_img: MatLike, alpha: float=0.8, beta: float=1.0, lambd: float=0.0):
    return cv2.addWeighted(initial_img, alpha, img, beta, lambd)

def mask_white_yellow(image: MatLike):
    converted = convert_hls(image)

    # Isolating white lines
    lower = np.array([50,   140, 0  ], dtype=np.uint8)
    upper = np.array([255, 255, 140], dtype=np.uint8)
    white_mask_1 = cv2.inRange(converted, lower, upper)

    lower = np.array([0,   130, 0  ], dtype=np.uint8)
    upper = np.array([255, 255, 40], dtype=np.uint8)
    white_mask_2 = cv2.inRange(converted, lower, upper)
    white_mask = cv2.bitwise_or(white_mask_1, white_mask_2)


    # Isolating yellow lines
    lower = np.array([10,   0, 60], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(converted, lower, upper)

    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    return yellow_image, white_image


def get_top_y_point(line_params: list[float], y_min: int):
    a_left  = line_params[0]
    b_left  = line_params[1]
    a_right = line_params[2]
    b_right = line_params[3]

    if a_right - a_left == 0:
        y_intersection = y_min
    else:
        y_intersection = int((a_right * b_left - a_left * b_right) / (a_right - a_left))
    margin = 10
    if (y_intersection + margin > y_min):
        y_min = y_intersection + margin

    return y_min

def draw_lanes(image: MatLike) -> tuple[tuple[int, int, int, int], MatLike]:
    h, w = image.shape[0], image.shape[1]
    
    yellow_image, white_image = mask_white_yellow(image)
    yellow_lines_left = 0
    yellow_lines_right = 0
    white_lines_left = 0
    white_lines_right = 0
    orig_with_found_lanes = np.copy(image)
    for color_idx, masked_image_ in enumerate([yellow_image, white_image]):
        if color_idx == 0:
            line_color = (255, 255, 0)
        else:
            line_color = (0, 255, 255)
        gray_image = grayscale(masked_image_)
        blurred_image = gaussian_blur(gray_image, 5)
        edges_image = canny(blurred_image, 40, 80)
        y_top_mask = h * 0.55
        y_bot_mask = h * 0.95
        vertices = np.array([
            [0, y_bot_mask], [w * 0.25, y_top_mask], [w * 0.75, y_top_mask], [w, y_bot_mask]
        ], dtype=np.int32)
        masked_image = region_of_interest(edges_image, [vertices])

        rho = 1
        theta = np.pi / 180
        threshold = 30
        min_line_length = 50
        max_line_gap = 5

        hough_lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

        lines_filtered = []
        c_len_left = 0
        c_len_right = 0
        left_hough_lines_exists = False
        right_hough_lines_exists = False
        a_left = 0
        b_left = 0
        a_right = 0
        b_right = 0
        y_min = 10000

        if hough_lines is not None :
            for line in hough_lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 == 0:
                        a = float('inf')
                    else:
                        a = float((y2 - y1)/(x2 - x1))
                    b = (y1 - a * x1)
                    length = math.sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2))

                    if not (np.isnan(a) or np.isinf(a) or a == 0):
                        if (a > -1.6) and (a < -0.35):
                            if y1 < y_min:
                                y_min = y1
                            if y2 < y_min:
                                y_min = y2
                            if color_idx == 0:
                                yellow_lines_left += 1
                            else:
                                white_lines_left += 1
                            lines_filtered.append(line)
                            c_len_left += pow(length, 2)
                            left_hough_lines_exists = True
                            a_left += a * pow(length, 2)
                            b_left += b * pow(length, 2)

                        if (a > 0.35) and (a < 1.6):
                            if y1 < y_min:
                                y_min = y1
                            if y2 < y_min:
                                y_min = y2
                            if color_idx == 0:
                                yellow_lines_right += 1
                            else:
                                white_lines_right += 1
                            lines_filtered.append(line)
                            c_len_right += pow(length, 2)
                            left_hough_lines_exists = True
                            a_right += a * pow(length, 2)
                            b_right += b * pow(length, 2)

        y_max = h

        if c_len_left != 0:
            a_left /= c_len_left
            b_left /= c_len_left

        if c_len_right != 0:
            a_right /= c_len_right
            b_right /= c_len_right

        line_params = [a_left, b_left, a_right, b_right]

        y_min = get_top_y_point(line_params, y_min)

        x1_left = 0
        x2_left = 0
        x1_right = 0
        x2_right = 0

        a_left  = line_params[0]
        b_left  = line_params[1]
        a_right = line_params[2]
        b_right = line_params[3]

        if a_left != 0:
            x1_left = int((y_max - b_left) / a_left)
            x2_left = int((y_min - b_left) / a_left)

        if a_right != 0:
            x1_right = int((y_max - b_right) / a_right)
            x2_right = int((y_min - b_right) / a_right)

        found_lines_image = np.zeros((h, w, 3), dtype=np.uint8)

        if a_left != 0:
            cv2.line(found_lines_image, (x1_left, y_max), (x2_left, y_min), line_color, 7)

        if a_right != 0:
            cv2.line(found_lines_image, (x1_right, y_max), (x2_right, y_min), line_color, 7)

        orig_with_found_lanes = weighted_img(found_lines_image, orig_with_found_lanes)

    return (yellow_lines_left, yellow_lines_right, white_lines_left, white_lines_right), orig_with_found_lanes

def predict_road_lines(image: MatLike) -> tuple[dict[str, float], MatLike]:
    """Predicts country probabilities based on the combination of road line colors and their positions in the image."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (L_yellow, R_yellow, L_white, R_white), res_img = draw_lanes(rgb_image)
    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    total_lines = L_yellow + R_yellow + L_white + R_white + 1e-6
    features = {
        'left_yellow_ratio': L_yellow / total_lines,
        'left_white_ratio': L_white / total_lines,
        'right_yellow_ratio': L_yellow / total_lines,
        'right_white_ratio': L_white / total_lines,
        'left_vs_right_yellow': (L_yellow - R_yellow) / total_lines,
        'left_vs_right_white': (L_white - R_white) / total_lines,
    }
    scores = []
    for c, profile in COUNTRY_PROFILES.items():
        # Simple L2 distance
        dist = np.linalg.norm([features[k] - profile[k] for k in features])
        # Convert distance to likelihood (the smaller the better)
        likelihood = np.exp(-dist)
        scores.append((c, likelihood))

    total = sum(l for _, l in scores)
    return {c: l / total for c, l in scores}, res_img

