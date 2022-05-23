import cv2
import numpy as np

def blob_detection(self, image):
    # Transform image into HSV, select parts within the predefined red range color as a mask,
    # dilate and erode the selected parts to remove noise
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask_dilation = cv2.dilate(mask, kernel, iterations=2)
    mask_final = cv2.erode(mask_dilation, kernel, iterations=1)
    kernel = np.ones((6, 6), np.float32) / 25
    mask_final = cv2.filter2D(mask_final, -1, kernel)

    # Apply mask to original image, show results
    res = cv2.bitwise_and(image, image, mask=mask_final)
    cv2.imshow('mask', mask_final)
    cv2.imshow('image seen through mask', res)

    # Parameter definition for SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 200000
    params.filterByInertia = True
    params.minInertiaRatio = 0.0
    params.maxInertiaRatio = 0.8

    # params.filterByConvexity = True
    # params.minConvexity = 0.09
    # params.maxConvexity = 0.99

    # Applying the params
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(~mask_final)

    # draw
    im_with_keypoints = cv2.drawKeypoints(~mask_final, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)

    ## Find outer contours
    im, contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Fidining the Max contour
    maxContour = 0
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour

    ## Draw
    cv2.drawContours(image, maxContourData, -1, (0, 255, 0), 2, lineType=cv2.LINE_4)
    cv2.imshow('image with countours', image)

    # Calculate image moments of the detected contour
    M = cv2.moments(maxContourData)

    try:
        # Draw a circle based centered at centroid coordinates
        xPixel = int(M['m10'] / M['m00'])
        yPixel = int(M['m01'] / M['m00'])
        cv2.circle(image, ( xPixel, yPixel ), 5, (0, 0, 0), -1)
        rospy.loginfo("Biggest blob: x Coord: " + str(xPixel) + " y Coord: " + str(yPixel) + " Size: " + str(blobSize))

        # Show image:
        cv2.imshow("outline contour & centroid", image)

    except ZeroDivisionError:
        pass

    # Save center coordinates of the blob as a Point() message

    # blob_coordinates_msg = Point(int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), 0)

    # blob_coordinates_msg.x = int(M['m10'] / M['m00'])
    # blob_coordinates_msg.y = int(M['m01'] / M['m00'])
    # blob_coordinates_msg.z = 0

    # Publish center coordinates
    # self.redBlobPub.publish(blob_coordinates_msg)