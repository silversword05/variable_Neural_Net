import cv2
import glob

img_list = glob.glob("test/*")
img_list = sorted(img_list)

no_of_img = 500

f = open("data.txt","w")
for i in range(no_of_img):
    img = cv2.imread(img_list[i], 0)
    mat = img.reshape(784)
    dat = ""
    for j in range( len(mat) ):
        dat = dat + str(mat[j]) +" "
    dat = dat + str( int( img_list[i][-5] ) ) + "\n"
    f.write(dat)

f.close()