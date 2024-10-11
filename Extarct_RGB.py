import cv2
import numpy as np
import csv

file_title = "Cu_AH_T"

csvfile = '{}.csv'.format(file_title)

img = cv2.imread('{}.jpg'.format(file_title))

if img is None:
    print("No Pic")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

min_radius = 20
max_radius = 28
param1 = 20
param2 = 20
min_dist = 15

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_dist,
                            param1=param1, param2=param2,
                            minRadius=min_radius, maxRadius=max_radius)

all_radii = []

circle_data = []

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        all_radii.append(radius)

        circle_rgb = img[center[1], center[0]]
        circle_data.append(list(circle_rgb))

if all_radii:
    avg_radius = sum(all_radii) / len(all_radii)
    print(f"average R: {avg_radius:.2f}")
else:
    print("No detection")

with open(csvfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Red', 'Green', 'Blue'])
    writer.writerows(circle_data)

cv2.imshow('Detected Circles', img)
cv2.imwrite('THough.tif', img)
cv2.waitKey(0)
cv2.destroyAllWindows()