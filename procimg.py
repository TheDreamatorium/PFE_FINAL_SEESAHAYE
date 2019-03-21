import cv2
import imutils
import numpy as np

class ProcessImage:

    #Constructor reads the image from the path provided
    def __init__(self, image):
        self.image = image
        self.characters = []
        self.gray = []
        self.thresh = []
        self.points = []
        self.output = []

    def prepareImg(self):
        #resize the width
        if self.image.shape[1] > 320:
            print(self.image.shape)
            self.image = imutils.resize(self.image,width=320)

        self.output = self.image
        #Turn into grayscale image
        self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

        self.gray = cv2.GaussianBlur(self.gray,(5,5),0,cv2.BORDER_REPLICATE)

        self.gray = cv2.copyMakeBorder(self.gray,top=10,bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT)

        return self.gray

    def detectDarkLightRegs(self):
        #Create a 5x5 structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

        #Blackhat morphological to detect dark objects(characters) on light background(paper)
        blackhat = cv2.morphologyEx(self.prepareImg(),cv2.MORPH_BLACKHAT,kernel)

        # cv2.imshow("After blackhat", blackhat)
        # cv2.waitKey(0)

        _,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_OTSU)

        #Dilation is used to expand the 1s in the binary image in order to improve accuracy during recognition
        
        thresh = cv2.dilate(thresh,None)

        # #LINK: https://gist.github.com/Ankita-Das/82bce39b35d1bbeca2ce87c4e8aba33d (DESKEWING)
        # coords = np.column_stack(np.where(thresh > 0))
        # angle = cv2.minAreaRect(coords)[-1]

        # if angle < -45:
	    #     angle = -(90 + angle)

        # else:
	    #     angle = -angle


        # (h, w) = self.gray.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # thresh = cv2.warpAffine(thresh, M, (w, h),
        #     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


        # #Erode the picture to reduce defects create by dilation
        # thresh = cv2.erode(thresh, None)

        self.thresh = thresh

        # cv2.imshow("thresh", self.thresh)
        # cv2.waitKey(0)

        return self.thresh

    def findCnts(self):
        img = self.detectDarkLightRegs()
        _,cnts,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        return cnts

    def extractCharacters(self):
        cnts = self.findCnts()

        #Calculate the average area of the contours
        avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])
        thresh = self.thresh
        self.characters = []

        for (i,c) in enumerate(cnts):
            
            # if the area is less than 20% than the average then skip
            if cv2.contourArea(c) > avgCntArea * 0.20:
                #create a mask of size 28 x 28 (same as dataset images)
                mask = np.zeros(self.gray.shape,dtype="uint8")
                
                (x,y,w,h) = cv2.boundingRect(c)
                hull = cv2.convexHull(c)

                cv2.drawContours(mask,[hull],-1,255,-1)
                mask = cv2.bitwise_and(thresh,thresh,mask=mask)
                
                char = mask[y-8:y+h+8,x-8:x+w+8]


                coords = np.column_stack(np.where(char > 0))
                angle = cv2.minAreaRect(coords)[-1]

                if angle < -45:
	                angle = -(90 + angle)

                else:
	                angle = -angle


                (h, w) = char.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                char = cv2.warpAffine(char, M, (w, h),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                if np.size(char) > 0:
                    print(cv2.contourArea(c))
                    # cv2.imshow("char", char)
                    # cv2.waitKey(0)
                    char = cv2.resize(char,(28,28))
                    self.characters.append(char)
                    self.points.append((x,y,w,h))
            
            
        return self.characters, self.points
