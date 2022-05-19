from collections import defaultdict
#from goto import goto, comefrom, label
import cv2
import math
import numpy as np


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections




'''Erosion'''
def filter_E(mask):
    kernel = np.ones((7,7), np.uint8)                      #np.uint8 = Byte (-128 to 127) : black white color range
    result = cv2.erode(mask,kernel,iterations = 1)          #morphological transformation with mask
    return result

'''Dilation'''
def filter_ED(mask):
    kernel = np.ones((7,7), np.uint8)                      #np.uint8 = Byte (-128 to 127) : black white color range
    #https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    erode = cv2.erode(mask,kernel,iterations = 1)          #morphological transformation with mask
    result = cv2.dilate(erode, kernel,iterations = 2)      #since the white noise removed, the size is small but dialatation is needed
    return result

'''Gaussian blur'''
def filter_EDG(mask):
    kernel = np.ones((7,7), np.uint8)                      #np.uint8 = Byte (-128 to 127) : black white color range
    #https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    erode = cv2.erode(mask,kernel,iterations = 1)          #morphological transformation with mask
    dilate = cv2.dilate(erode, kernel,iterations = 2)      #since the white noise removed, the size is small but dialatation is needed
    result = cv2.GaussianBlur(dilate,(5,5),2)              
    return result


'''Distance calculate between two points'''
def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

'''Hough Line Analysis'''
def RT_plane(image1, GB, THR1 = 50, THR2 = 150):
    print("Rho-Theta plane generating")
    img = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(5,5),GB)
    Can1 = cv2.Canny(blur,THR1,THR2,apertureSize = 3)
    
    
    
    iH, iW = Can1.shape
    distMax = round( math.sqrt( (iH**2 + iW**2) ) )
    
    theta = np.arange(-90,89)
    rho = np.arange(-distMax,distMax,1)
    
    H = np.zeros(( len(rho), len(theta) ))
    
    for ix in range(iW):
        for iy in range(iH):
            if (Can1[iy][ix] != 0):
                
                for iTheta in range(len(theta)):
                    t = theta[iTheta] * math.pi/180
                    
                    dist = int( ix*np.cos(t) + iy*np.sin(t) )
                    
                    #print(dist)
                    #print(rho)
                    #print(rho - dist)
                    
                    
                    #print( abs(rho - dist) )
                    #print( min(abs(rho - dist)) )
                    
                    #d, iRho = min(abs(rho - dist))
                    
                    A = rho - dist
                    
                    d = A[0]
                    iRho = A[1]
                    
                    if (d <= 1):
                        H[iRho][iTheta] = H[iRho][iTheta] + 1
    return H

'''Hough Line Transform & Line Intersections'''
def HL_IS(image1, GB, Threshold, Theta, THR1 = 50, THR2 = 150):
    img = image1
    gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),GB)
    HL_IS.Canny = cv2.Canny(blur,THR1,THR2,apertureSize = 3)
    
    lines = cv2.HoughLines(HL_IS.Canny,1,np.pi/180*Theta,Threshold)
    
    if lines is None : 
        print("No line Detected!") #if there's no line generated
        HL_IS.ISM = False
        return True  #Break the function
        
    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)
    
    if len(intersections) == 0 : 
        print("Break since no intersection")
        return True
    
    print("lines Iintersections : " + str( len(intersections) ))
    
    """Line Drawing"""
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)

    """Intersection Point Drawing"""
    pt_Number = 0
    for pt in intersections:
        pt_Number += 1
        IS = pt[0]
        IS = tuple(IS)
        print("IS coordinate #" + str(pt_Number) + " : " + str(IS))
        cv2.circle(img, IS, 3, (125, 0, 125), 2)
    
    ISLine_coordinate_mean = np.mean(intersections, axis = 0, dtype=np.int_)
    #meaning all elements in the axises vertically
    
    HL_IS.ISM = tuple(ISLine_coordinate_mean[0]) #Assigning coordinate value to transfer to Bluetooth
    print("IS coordinate's mean : " + str( HL_IS.ISM))
    cv2.circle(img,  HL_IS.ISM, len(intersections), (0, 255, 255), 2) #Weed Center detection
    
    
    return img