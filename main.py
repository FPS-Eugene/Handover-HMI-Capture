print('Loading dependencies')
from asyncio.windows_events import NULL
#import tkinter as tk
#from tkinter.filedialog import askdirectory
#tk.Tk().withdraw()
from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/eugene.bodnarchuk/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
import pyautogui
import numpy as np
import cv2 as cv
import enchant
import re
print('Successfully loaded dependencies')

def manualScreenshotControl():
    import keyboard
    screenshot_Counter = 0
    while True:
        print('Manual Screenshot Mode. Press ";" to take a screenshot of the window')
        print('or type "k" for a numpad screenshot')
        print('or type "l" for a login screenshot')
        print('Type "e" to exit anytime')
        
        selectionEvent = keyboard.read_event()
        while selectionEvent.event_type == 'down':      #Wait for key realease
            selectionEvent = keyboard.read_event()
        selectionBuff = selectionEvent.name
        screenshot_Successful = False


        if selectionBuff == 'e':        #Exit Function
            return
        elif selectionBuff == ';':
            screenshotBuffer = pyautogui.screenshot(region=screenShotWindow)
            screenshot_Successful = True
            print('Took Window Screenshot\n')
        elif selectionBuff == 'k':
            screenshotBuffer = pyautogui.screenshot(region=keypadWindow)
            screenshot_Successful = True
            print('Took KeypadScreenshot\n')
        elif selectionBuff == 'l':
            screenshotBuffer = pyautogui.screenshot(region=loginWindow)
            screenshot_Successful = True
            print('Took Login Scr5eenshot\n')
        else:
            print('Unknown command...\n')

        if screenshot_Successful:
            screenshot_Counter += 1
            screenshotBuffer.save(rf"{screenshot_Path}/{screenshot_Counter}.png")   
def automaticScreenshotControl():
    import time
    print('Pre-defined Screenshot Mode. Press Enter to take a screenshot of the window')
    
    imageCounter = 0
    for i in range(len(actionList)):
        buttonIndex = actionList[i]['buttonIndex']
        takeScreenshot = actionList[i]['takeScreenshot']

        pyautogui.click(buttonInfo[buttonIndex]['guiPos'][0], buttonInfo[buttonIndex]['guiPos'][1])
        time.sleep(0.5)
        if takeScreenshot:
            time.sleep(0.75) 
            screenshotBuffer = pyautogui.screenshot(region=screenShotWindow)
            #screenshotBuffer.save(rf"{screenshot_Path}/{buttonInfo[buttonIndex]['name']}.png")
            screenshotBuffer.save(rf"{screenshot_Path}/{imageCounter}.png")
            imageCounter += 1
            time.sleep(0.25)  


# function prepareImage
# input: img = PIL image, threshhold = int 0-255
# output: openCV image
def prepareImage(img: Image.Image, threshhold: int):
    funcImg = ImageOps.invert(img)
    funcImg = ImageOps.grayscale(funcImg)
    funcImg = funcImg.resize((buttonWindow[2] * 2, buttonWindow[3] * 2), resample=Image.Resampling.LANCZOS)
    # img.show()
    imgarray = np.array(funcImg)
    # img1 = 255 / (np.power(math.e, -0.1 * (img1 - 100)) + 1)
    imgarray = np.where(imgarray > threshhold, 255, 0)
    funcImg.putdata(imgarray.flatten())
    return np.array(funcImg)
# debug function: Shows text bounding boxes
def showBoundingBox(img: np.ndarray):
    tempData = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(tempData['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (tempData['left'][i], tempData['top'][i], tempData['width'][i], tempData['height'][i])
        cv.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 2)
    cv.imshow('img', img)
    cv.waitKey()

def findText(originalimg: Image.Image):
    # Prepare Vars
    threshholdResults = []
    # Resize, invert, and greyscale the image
    baseImg = ImageOps.grayscale(ImageOps.invert(originalimg)).resize((buttonWindow[2] * 4, buttonWindow[3] * 4), resample=Image.Resampling.LANCZOS)
    baseImgarray = np.array(baseImg)
    print('Starting Refinement Loop')
    # Run a dirty first pass to get an idea of the most optimal settings, then run a fine pass
    for i in range(50, 201, 25):  # Quick 25 increment sweep
        FL_confidence = 0
        FL2_confidence = 0
        FL_ImgArr = np.where(baseImgarray > i, 255, 0).astype('uint8')
        FL_ImgData = pytesseract.image_to_data(FL_ImgArr, output_type=pytesseract.Output.DICT, config='--psm 10')
        for j in range(len(FL_ImgData['conf'])):
            FL2_confidence = FL_ImgData['conf'][j]
            if FL2_confidence != -1:                    # Skip missing content
                FL_confidence += FL2_confidence - 50    # Any confidense below 50 is considered harmful
        FL_text = "".join(FL_ImgData['text'])   # Conver list of letters into a single string
        threshholdResults.append({'confidence': FL_confidence, 'threshhold': i, 'text': FL_text})
    threshholdResults = sorted(threshholdResults, key=lambda k: k['confidence'], reverse = True)    #Sort by most confident answer
    print("Finished corse pass. Starting medium pass")
    
    lowTHStart = threshholdResults[0]['threshhold'] - 25    # Set the lower and upper bounds for the meduim scan
    highTHStart = threshholdResults[0]['threshhold'] + 25
    threshholdResults = []
    for i in range(lowTHStart, highTHStart, int((highTHStart - lowTHStart) / 4)):  # 4 passes of medium tolarance
        FL_confidence = 0
        FL2_confidence = 0
        FL_ImgArr = np.where(baseImgarray > i, 255, 0).astype('uint8')
        FL_ImgData = pytesseract.image_to_data(FL_ImgArr, output_type=pytesseract.Output.DICT)
        for j in range(len(FL_ImgData['conf'])):
            FL2_confidence = FL_ImgData['conf'][j]
            if FL2_confidence != -1:
                FL_confidence += FL2_confidence - 50
        FL_text = sorted(FL_ImgData['text'], key=len, reverse = True)[0]   # At this stage, the longest single word is probably what we're looking for
        threshholdResults.append({'confidence': FL_confidence, 'threshhold': i, 'text': FL_text})
    threshholdResults = sorted(threshholdResults, key=lambda k: k['confidence'], reverse = True)
    print("Finished medium pass. Starting fine pass")
    
    lowTHStart = threshholdResults[0]['threshhold'] - 5    # Set the lower and upper bounds for the fine scan
    highTHStart = threshholdResults[0]['threshhold'] + 5
    threshholdResults = []
    for i in range(lowTHStart, highTHStart, 2):  # 2 increment sweep
        FL_confidence = 0
        FL2_confidence = 0
        FL_ImgArr = np.where(baseImgarray > i, 255, 0).astype('uint8')
        FL_ImgData = pytesseract.image_to_data(FL_ImgArr, output_type=pytesseract.Output.DICT)
        for j in range(len(FL_ImgData['conf'])):
            FL2_confidence = FL_ImgData['conf'][j]
            if FL2_confidence != -1:
                FL_confidence += FL2_confidence - 50
        FL_text = sorted(FL_ImgData['text'], key=len, reverse = True)[0]   # At this stage, the longest single word is probably what we're looking for
        threshholdResults.append({'confidence': FL_confidence, 'threshhold': i, 'text': FL_text})
    threshholdResults = sorted(threshholdResults, key=lambda k: k['confidence'], reverse = True)
    print("Finished fine pass.")

    results = re.sub(r'[^\w]', '', threshholdResults[0]['text'])
    print(results)
    # For debugging
    buffImg = baseImg
    FL_ImgArr = np.where(baseImgarray > threshholdResults[0]['threshhold'], 255, 0).astype('uint8')
    buffImg.putdata(FL_ImgArr.flatten())
    buffImg.show()

    return results
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
def find_squares(img: np.ndarray):
    xrange = range
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 2500 and cv.contourArea(cnt) < 35000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

# TODO Implement button recognition
# Done finish reliable text recognition
# TODO Implement tree type gui exporation. 
# TODO Use image similarity tests to recognize loops and determine branches







# if dictionary.check(text1):
#     print(text1)
# elif dictionary.check(text2):
#     print(text2)
# else:
#     print(f'No word found in text1 or text2: {text1} ᓚᘏᗢ {text2}')
# print('Press Enter for manual triggering or Type 1 and press Enter for pre-defined screenshots')
# selectionBuff = input()

# if selectionBuff == '':
#     manualScreenshotControl()

# elif selectionBuff == '1':
#     automaticScreenshotControl()


if __name__ == '__main__':
    screenShotWindow = [0,0,800,600]        # Region Selection (xstart, ystart, width, height)
    buttonWindow = [4,20,94,46]
    keypadWindow = [775, 180, 370, 720]     
    loginWindow = [720, 430, 480, 220]      
    dictionary = enchant.Dict("en_US")

    # print('Select folder to save screenshots')
    # screenshot_Path = askdirectory()
    screenshot_Path = "C:/Users/eugene.bodnarchuk/OneDrive - FPS Food Process Solutions Corporation/Documents/Training/81390/AutoScreenshot"
    filename = 'C:/Users/eugene.bodnarchuk/OneDrive - FPS Food Process Solutions Corporation/Documents/Test Ideas/Auto_Screenshot/ocrTest.jpg'
    # img1 = np.array(Image.open(filename))

    print('Capturing')
    screenshotBuffer = pyautogui.screenshot(region=screenShotWindow)

    
    
    open_cv_image = np.array(screenshotBuffer)[:, :, ::-1].copy() 
    squares = find_squares(open_cv_image)
    cv.drawContours( open_cv_image, squares, -1, (0, 255, 0), 3 )
    cv.imshow('squares', open_cv_image)
    ch = cv.waitKey()

    buttonBuffer = pyautogui.screenshot(region=buttonWindow)
    findText(buttonBuffer)

    print('Done. Press Enter to exit')
    input()
    exit()










