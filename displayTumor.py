import numpy as np
import cv2 as cv

class DisplayTumor:
    def __init__(self):
        self.image = None
        self.thresh = None  # Initialize thresh attribute
        self.kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological operations

    def readImage(self, image):
        self.image = np.array(image)  # Convert PIL image to NumPy array
        self.preprocessImage()

    def preprocessImage(self):
        # Convert to grayscale
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, self.thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    def removeNoise(self):
        if self.thresh is None:
            raise ValueError("Threshold image (thresh) is not set. Call preprocessImage() first.")

        # Perform morphological opening to remove noise
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        return opening

    def displayTumor(self):
        # Example method to display the tumor region
        if self.thresh is None:
            raise ValueError("Threshold image (thresh) is not set. Call preprocessImage() first.")

        # Find contours
        contours, _ = cv.findContours(self.thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv.drawContours(self.image, [contour], -1, (0, 255, 0), 2)

        return self.image