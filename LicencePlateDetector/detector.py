import cv2
import easyocr
import os
import matplotlib.pyplot as plt

# Create an OCR reader using the easyocr library, specify the language as Romanian.
reader = easyocr.Reader(['ro'])

def extract_license_plate(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to gray scale as it's easier to process
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use a Haar Cascade to detect license plates in the image.
    # Note: The current cascade is for Russian plates. You need a suitable one for Romanian plates.
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = plate_cascade.detectMultiScale(gray, 1.1, 10)

    # For each detected license plate
    for (x, y, w, h) in plates:
        # Extract the region of interest (ROI) i.e., the license plate
        roi_color = img[y:y + h, x:x + w]

        # Use the easyocr reader to read the text in the ROI
        result = reader.readtext(roi_color)

        # Each result contains the coordinates of the text and the text itself
        for detection in result:
            top_left = tuple([int(val) for val in detection[0][0]])
            bottom_right = tuple([int(val) for val in detection[0][2]])
            text = detection[1]
            print(f'Image: {image_path}, Detected Text: {text}, Top Left: {top_left}, Bottom Right: {bottom_right}')

        # Draw a rectangle around the license plate in the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Create a figure for the two images
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Show the license plate in the first subplot
        ax[0].imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        ax[0].set_title('License Plate')

        # Show the original image with a rectangle around the license plate in the second subplot
        ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Original Image')

        # Display the figure
        plt.show()



# Directory containing your images
image_directory = 'images'

# Get a list of all the image files in the directory
image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith(".jpg")]

# Process each image file
for image_file in image_files:
    extract_license_plate(image_file)
