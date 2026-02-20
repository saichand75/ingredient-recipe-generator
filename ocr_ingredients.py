import cv2
import pytesseract
import re

# --------------------------------------
# 1. Set Tesseract Path (Windows Only)
# --------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------
# 2. Load Image
# --------------------------------------
image = cv2.imread("package.jpg")

if image is None:
    print("Error: package.jpg not found!")
    exit()

# --------------------------------------
# 3. Preprocess Image (Improve OCR)
# --------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

# --------------------------------------
# 4. Extract Full Text
# --------------------------------------
text = pytesseract.image_to_string(gray)

print("\n----- FULL EXTRACTED TEXT -----\n")
print(text)

# --------------------------------------
# 5. Extract Ingredients Section
# --------------------------------------
match = re.search(r"ingredients[:\-]?\s*(.*)", text, re.IGNORECASE)

if match:
    ingredients_text = match.group(1)

    # Clean and split into list
    ingredients_list = [item.strip() for item in ingredients_text.split(",")]

    print("\n----- DETECTED INGREDIENTS -----\n")
    for ingredient in ingredients_list:
        print("-", ingredient)
else:
    print("\nIngredients section not found.")