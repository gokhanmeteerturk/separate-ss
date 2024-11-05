import os
import shutil
from PIL import Image
import numpy as np
import cv2

# Define directories
detected_dir = 'detected-screenshots'
undetected_dir = 'undetected'

# Create directories if they don't exist
os.makedirs(detected_dir, exist_ok=True)
os.makedirs(undetected_dir, exist_ok=True)

def load_image_files(directory):
    """Load image files from the current directory."""
    image_files = [file for file in os.listdir(directory) if file.endswith(('.jpeg', '.jpg', '.png'))]
    return image_files

def classify_colors(image_path, resize_to=(100, 100)):
    """Classify pixels into color categories based on HSV values."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resize_to)  # Downsample for efficiency
    image = np.array(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define HSV thresholds
    # White/Grey: High brightness, low saturation
    white_mask = (hsv_image[:, :, 1] < 40) & (hsv_image[:, :, 2] > 200)

    # Blue/Green: Specific hue ranges
    blue_mask = ((hsv_image[:, :, 0] >= 100) & (hsv_image[:, :, 0] <= 140)) & (hsv_image[:, :, 1] > 50)
    green_mask = ((hsv_image[:, :, 0] >= 60) & (hsv_image[:, :, 0] <= 90)) & (hsv_image[:, :, 1] > 50)

    # Other colors (excluding near-black)
    other_mask = ~white_mask & ~blue_mask & ~green_mask & (hsv_image[:, :, 2] > 40)

    # Calculate percentages
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
    white_percentage = np.sum(white_mask) / total_pixels * 100
    blue_green_percentage = (np.sum(blue_mask) + np.sum(green_mask)) / total_pixels * 100
    other_percentage = np.sum(other_mask) / total_pixels * 100

    return white_percentage, blue_green_percentage, other_percentage

def is_imessage_screenshot(white_percentage, blue_green_percentage, other_percentage, file_name):
    """
    Determine if an image meets the criteria for being an iMessage screenshot,
    with an additional rule for high white/grey percentages.
    """
    # Print debug information
    print(f"\nFile: {file_name}")
    print(f"White/Grey %: {white_percentage:.2f}")
    print(f"Blue/Green %: {blue_green_percentage:.2f}")
    print(f"Other Colors %: {other_percentage:.2f}")

    # Main criteria
    if white_percentage >= 35 and blue_green_percentage >= 8 and other_percentage <= 23:
        print("Classified as iMessage screenshot.")
        return True
    # Additional condition for high white percentage
    elif white_percentage > 70 and other_percentage <= 20:
        print("Classified as iMessage screenshot (high white/grey percentage).")
        return True
    else:
        print("Classified as not an iMessage screenshot.")
        return False


def classify_images(directory):
    """Classify images into iMessage screenshots or other."""
    for file_name in load_image_files(directory):
        file_path = os.path.join(directory, file_name)
        white_percentage, blue_green_percentage, other_percentage = classify_colors(file_path)

        # Classify based on the criteria
        if is_imessage_screenshot(white_percentage, blue_green_percentage, other_percentage, file_name):
            shutil.copy(file_path, os.path.join(detected_dir, file_name))
        else:
            shutil.copy(file_path, os.path.join(undetected_dir, file_name))

if __name__ == "__main__":
    current_directory = os.getcwd()
    classify_images(current_directory)
