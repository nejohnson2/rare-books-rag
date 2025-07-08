# Save this as debug_ocr.py or run in a Jupyter notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

# Set this to your Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\nijjohnson\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def show_image(title, img):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def debug_image_processing(image_path):
    # Load original
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    show_image("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale", gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    show_image("Denoised", denoised)

    # Threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 101, 2
    )
    show_image("Thresholded", thresh)

    # Deskew (simple version)
    coords = np.column_stack(np.where(thresh < 255))
    angle = 0
    if coords.shape[0] > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = thresh
    show_image(f"Deskewed (angle={angle:.2f})", deskewed)

    # OCR
    print("\n--- Raw OCR Output ---")
    text = pytesseract.image_to_string(deskewed, lang='eng')
    print(text)

    print("\n--- OCR Data ---")
    data = pytesseract.image_to_data(deskewed, lang='eng', output_type=pytesseract.Output.DICT)
    for i in range(len(data['text'])):
        print(f"Word: '{data['text'][i]}' | Conf: {data['conf'][i]}")

# Example usage
debug_image_processing(r".\images\unnamed.jpg")