# Image Scanning

1. edge detection

   <img src="./img/edgeDetection.jpg" alt="edgeDetection" style="zoom:60%;" />

2. find contour

   <img src="./img/findContour.jpg" alt="findContour" style="zoom: 60%;" />

3. perspective transform

   <img src="./img/perspectiveTransform.jpg" alt="perspectiveTransform" style="zoom:8%;" />

4. result

   - situation 1

     <img src="./img/res1.JPEG" alt="res1" style="zoom:40%;" />

   - situation 2

     <img src="./img/res2.JPEG" alt="res2" style="zoom:40%;" />

   - situation 3

     <img src="./img/res3.JPEG" alt="res3" style="zoom:35%;" />

   - situation 4

     <img src="./img/res4.JPEG" alt="res4" style="zoom:40%;" />

   - situation 5

     <img src="./img/res5.JPEG" alt="res5" style="zoom:40%;" />

# Image Stitching

Design for screenshot stitching (delete duplicated area)

steps

1. set the minimum height of duplicated area
2. turn input images to gray-scale map
3. compare two images to find the location of duplicated area
   - like two sliding windows (the black rectangle)
   - for image 1, bottom up
   - for image 2, top down
   - find the bottom location
4. find the top location
5. crop the duplicated area and stitch the screenshots

### example

input screenshot

<img src="./img/screenshot.JPEG" alt="screenshot" style="zoom: 50%;" />

after stitching

<img src="./img/stitchRes.jpg" alt="stitchRes" style="zoom: 33%;" />