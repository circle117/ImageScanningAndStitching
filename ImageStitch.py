import imutils
import cv2
import numpy as np
import time


def image_stitch_normal(photo_list):
    """
    stitch without cropping the duplicated part
    """
    image_stitch = cv2.imread(photo_list[0])
    for i in range(1, len(photo_list)):
        photo = cv2.imread(photo_list[i])
        # resize
        if image_stitch.shape[1]>photo.shape[1]:
            photo = imutils.resize(photo, width= image_stitch.shape[1])
        else:
            image_stitch = imutils.resize(image_stitch, width=photo.shape[1])
        # stitch
        image_stitch = cv2.vconcat([image_stitch, photo])
    cv2.imwrite('test_1.jpg', image_stitch)


def image_stitch_auto(photo_list, height):
    """
    delete duplicated area and stitch images
    height: the minimum height of duplicated area
    ratio: resize ratio for accelerating the calculation
    """
    image_stitch = cv2.imread(photo_list[0])
    for i in range(1,len(photo_list)):
        image = cv2.imread(photo_list[i])

        # get gray-scale map, contract images
        image_stitch_gray = cv2.cvtColor(image_stitch, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        flag = False
        for loc1 in range(image_stitch_gray.shape[0]-1, height-2, -1):          # bottom up for image 1
            temp1 = image_stitch_gray[loc1-height+1:loc1+1, 0:image_stitch_gray.shape[1]]

            for loc2 in range(0,int(image_gray.shape[0]/2)):                    # top down for image 2
                temp2 = image_gray[loc2:loc2+height, 0:image_gray.shape[1]]

                if (temp1==temp2).all():
                    print(loc1,loc2)
                    cut_height = height
                    loc1-=height
                    loc2-=1

                    # find maximum duplicated area
                    while loc1!=-1:
                        if (image_stitch_gray[loc1]==image_gray[loc2]).all():
                            loc1-=1
                            loc2-=1
                            cut_height+=1
                        else:
                            break
                    # crop and stitch
                    image_stitch = image_stitch[0:loc1,0:image_stitch.shape[1]]
                    image = image[loc2+1:image.shape[0],0:image.shape[1]]
                    image_stitch = cv2.vconcat([image_stitch,image])
                    flag = True
                    break
            if flag:
                break
        # if there is no duplicated area
        else:
            image_stitch = cv2.vconcat([image_stitch, image])
            
    cv2.imwrite('./img/stitchRes.jpg', image_stitch)


if __name__=="__main__":
    start = time.time()
    photo_list = ['./img/screenshot1.PNG', './img/screenshot2.PNG', './img/screenshot3.PNG']
    # image_stitch_normal(photo_list)
    image_stitch_auto(photo_list, 40)
    print(time.time()-start)