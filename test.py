import cv2
import pytesseract


if __name__ == '__main__':
    image = cv2.imread('plate.jpg')
    # image = cv2.resize(image, (100, 50))
    text = pytesseract.image_to_string(image, lang='tha')
    print(text)
    cv2.imshow('image', image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
