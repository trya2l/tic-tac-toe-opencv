import sys
import cv2 as cv
import numpy as np

def main(argv):
    
    default_file = 'smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread('/home/allan/Documents/3IL/IMAGES/tic-tac-toe-opencv/src/img/generated/imd/imd_2.png', cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    


    kernel = 2
    element = cv.getStructuringElement(cv.MORPH_RECT, (2*kernel + 1, 2*kernel + 1), (kernel, kernel))

    src = cv.erode(src, element)

    cv.imshow("detected circles", src)
    cv.waitKey(0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    gray = cv.medianBlur(gray, 5)

    canny = gray
    canny = cv.Canny(gray, 10, 200)
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, rows,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)
    
    canny = cv.cvtColor(canny,cv.COLOR_GRAY2BGR)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for ind,i in enumerate(circles[0, :]):
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    
    
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])