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

def spiralLoadButtonPosition():         #Not done filling in positions
    buttonInfo = []

    buttonInfo.append({'name': 'Overview', 'guiPos': [50, 42]})

    buttonInfo.append({'name': 'Conveyor-SpiralFreeze', 'guiPos': [150, 42]})
    buttonInfo.append({'name': 'Conveyor-Takeup', 'guiPos': [30, 300]})
    #ScreenshotInfo.append({'name': 'Conveyor-Manual', 'guiPos': [50, 570]})
    buttonInfo.append({'name': 'Conveyor-Infeed', 'guiPos': [770, 300]})

    buttonInfo.append({'name': 'Pumpdown_Defrost', 'guiPos': [250, 42]})
    buttonInfo.append({'name': 'Pumpdown_Defrost-Sequential_Defrost', 'guiPos': [130, 550]})
    #buttonInfo.append({'name': 'Pumpdown_Defrost-Pumpdown_Time', 'guiPos': [160, 350]})
    #buttonInfo.append({'name': 'Pumpdown_Defrost-Defrost_Delay_Time', 'guiPos': [160, 350]})
    buttonInfo.append({'name': 'Pumpdown_Defrost-Coil_1', 'guiPos': [725, 215]})
    buttonInfo.append({'name': 'Pumpdown_Defrost-Coil_2', 'guiPos': [725, 275]})
    buttonInfo.append({'name': 'Pumpdown_Defrost-Coil_3', 'guiPos': [725, 350]})
    buttonInfo.append({'name': 'Pumpdown_Defrost-Coil_4', 'guiPos': [725, 415]})

    buttonInfo.append({'name': 'Nighthold', 'guiPos': [350, 42]})
    buttonInfo.append({'name': 'Nighthold-Setup', 'guiPos': [740, 580]})
    buttonInfo.append({'name': 'Nighthold-Exit', 'guiPos': [330, 310]})

    buttonInfo.append({'name': 'Wash_CIP', 'guiPos': [450, 42]})

    buttonInfo.append({'name': 'Setup', 'guiPos': [550, 42]})

    buttonInfo.append({'name': 'Setup-Recipe_Prod_Select', 'guiPos': [95, 165]})
    buttonInfo.append({'name': 'Setup-Recipe_Prod_Select-Edit', 'guiPos': [715, 120]})
    buttonInfo.append({'name': 'Setup-Recipe_Prod_Select-Copy', 'guiPos': [725, 175]})

    buttonInfo.append({'name': 'Fans_Manual', 'guiPos': [95, 235]})
    buttonInfo.append({'name': 'Door_Bypass', 'guiPos': [95, 315]})
    buttonInfo.append({'name': 'Temperature_Trend', 'guiPos': [95, 385]})
    buttonInfo.append({'name': 'Operation_Log', 'guiPos': [95, 455]})

    buttonInfo.append({'name': 'CIP_Recipe_Select', 'guiPos': [270, 165]})
    buttonInfo.append({'name': 'CIP_Rinse_Recipe_Setpoint', 'guiPos': [425, 310]})
    buttonInfo.append({'name': 'CIP_Wash_Recipe_Setpoint', 'guiPos': [625, 310]})
    buttonInfo.append({'name': 'CIP_Sanatize_Recipe_Setpoint', 'guiPos': [425, 400]})
    buttonInfo.append({'name': 'CIP_Pasturize_Recipe_Setpoint', 'guiPos': [625, 400]})
    #ScreenshotInfo.append({'name': 'User_Login', 'guiPos': [395, 540]})
    buttonInfo.append({'name': 'CIP_Edit_Recipe', 'guiPos': [680, 540]})
    buttonInfo.append({'name': 'CIP_Edit_Recipe-Recipe_Copy', 'guiPos': [630, 280]})
    buttonInfo.append({'name': 'CIP_Edit_Recipe-Rinse', 'guiPos': [425, 420]})
    buttonInfo.append({'name': 'CIP_Edit_Recipe-Wash', 'guiPos': [625, 420]})
    buttonInfo.append({'name': 'CIP_Edit_Recipe-Sanatize', 'guiPos': [425, 510]})
    buttonInfo.append({'name': 'CIP_Edit_Recipe-Pasturize', 'guiPos': [625, 510]})

    buttonInfo.append({'name': 'Conveyors_Manual', 'guiPos': [270, 235]})
    buttonInfo.append({'name': 'Door_Heater', 'guiPos': [270, 315]})
    buttonInfo.append({'name': 'Fan_Trend', 'guiPos': [270, 385]})
    buttonInfo.append({'name': 'PLC_Input', 'guiPos': [270, 455]})

    buttonInfo.append({'name': 'Beltwash_Dryer', 'guiPos': [435, 165]})
    buttonInfo.append({'name': 'Beltwash_Trend', 'guiPos': [435, 385]})
    buttonInfo.append({'name': 'Startup_Setup', 'guiPos': [435, 455]})

    buttonInfo.append({'name': 'Factory', 'guiPos': [645, 315]})
    buttonInfo.append({'name': 'PLC_Time_Sync', 'guiPos': [645, 385]})
    buttonInfo.append({'name': 'PLC_Output', 'guiPos': [645, 455]})

    buttonInfo.append({'name': 'Alarm', 'guiPos': [640, 42]})
    buttonInfo.append({'name': 'Alarm-History', 'guiPos': [75, 565]})

    buttonInfo.append({'name': 'Return', 'guiPos': [50, 42]})
    buttonInfo.append({'name': 'keypadEsc', 'guiPos': [860, 825]})
    buttonInfo.append({'name': 'credentialsEsc', 'guiPos': [1130, 570]})
    buttonInfo.append({'name': 'Next', 'guiPos': [250, 42]})
    buttonInfo.append({'name': 'trendsNext', 'guiPos': [150, 42]})
    return buttonInfo
def loadActionList():
    #[button location in list if number]

    actionList = []
    actionList.append({'buttonIndex':0, 'takeScreenshot':1})    #overview
    actionList.append({'buttonIndex':1, 'takeScreenshot':1})    #conveyors
    actionList.append({'buttonIndex':2, 'takeScreenshot':1})    #left conveyors
    actionList.append({'buttonIndex':3, 'takeScreenshot':0})    #right
    actionList.append({'buttonIndex':3, 'takeScreenshot':1})    #right conveyors
    actionList.append({'buttonIndex':4, 'takeScreenshot':1})    #pumpdown_defrost
    actionList.append({'buttonIndex':5, 'takeScreenshot':1})    #sequential defrost
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':6, 'takeScreenshot':1})    #coil1
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next coil
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next coil
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next coil
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':10, 'takeScreenshot':1})   #nighthold
    actionList.append({'buttonIndex':11, 'takeScreenshot':1})   #nhsetup
    actionList.append({'buttonIndex':12, 'takeScreenshot':0})   #nhexit
    actionList.append({'buttonIndex':13, 'takeScreenshot':1})   #washcip
    actionList.append({'buttonIndex':14, 'takeScreenshot':1})   #setup
    actionList.append({'buttonIndex':15, 'takeScreenshot':1})   #prodReciSel
    actionList.append({'buttonIndex':16, 'takeScreenshot':1})   #recipEdit
    actionList.append({'buttonIndex':17, 'takeScreenshot':1})   #recipCopy
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':18, 'takeScreenshot':1})   #fanMan
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':19, 'takeScreenshot':1})   #doorByp
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':20, 'takeScreenshot':1})   #airTrenAirOn
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #airoff
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #suction
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':21, 'takeScreenshot':1})   #opLog
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':22, 'takeScreenshot':1})   #cipRecipSel
    actionList.append({'buttonIndex':23, 'takeScreenshot':1})   #rinsCurr
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':24, 'takeScreenshot':1})   #washCurr1
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #washCurr2
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #washCurr3
    actionList.append({'buttonIndex':45, 'takeScreenshot':1})   #washinfo
    actionList.append({'buttonIndex':49, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':49, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':25, 'takeScreenshot':1})   #sanatizCurr
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':26, 'takeScreenshot':1})   #pasturizCurr
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':27, 'takeScreenshot':1})   #edit
    actionList.append({'buttonIndex':28, 'takeScreenshot':1})   #copy
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':29, 'takeScreenshot':1})   #rinsEdit
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':30, 'takeScreenshot':1})   #washEdit1
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #washEdit1
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #washEdit1
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':31, 'takeScreenshot':1})   #sanatizEdit
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':32, 'takeScreenshot':1})   #pasturizEdit
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':14, 'takeScreenshot':0})   #setup
    actionList.append({'buttonIndex':33, 'takeScreenshot':1})   #conveyMan
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':34, 'takeScreenshot':1})   #doorHeat
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':35, 'takeScreenshot':1})   #fanTren
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #nextTren
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':36, 'takeScreenshot':1})   #plcinput
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':37, 'takeScreenshot':1})   #bwDryer
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':38, 'takeScreenshot':1})   #bTrend
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':39, 'takeScreenshot':1})   #startupSetup
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':40, 'takeScreenshot':1})   #factory
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':49, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':41, 'takeScreenshot':1})   #plcTimSyn
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':42, 'takeScreenshot':1})   #plcOutput
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':48, 'takeScreenshot':1})   #next
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':43, 'takeScreenshot':1})   #alarm
    actionList.append({'buttonIndex':44, 'takeScreenshot':1})   #history
    actionList.append({'buttonIndex':45, 'takeScreenshot':0})   #return
    actionList.append({'buttonIndex':1, 'takeScreenshot':0})   #return

    return actionList
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
    buttonInfo = spiralLoadButtonPosition()
    actionList = loadActionList()
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
def prepareImage(img, threshhold: int):
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
def showBoundingBox(img):
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
    # Resize and greyscale the image
    baseImg = ImageOps.invert(originalimg)
    baseImg = ImageOps.grayscale(baseImg)
    baseImg = baseImg.resize((buttonWindow[2] * 4, buttonWindow[3] * 4), resample=Image.Resampling.LANCZOS)
    baseImgarray = np.array(baseImg)
    print('Starting Refinement Loop')
    # Run a dirty first pass to get an idea of the most optimal settings, then run a fine pass
    for i in range(25, 225, 25):  # Quick 25 increment sweep
        FL_confidence = 0
        FL2_confidence = 0
        FL_ImgArr = np.where(baseImgarray > i, 255, 0)
        FL_ImgArr = FL_ImgArr.astype('uint8')
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
        FL_ImgArr = np.where(baseImgarray > i, 255, 0)
        FL_ImgArr = FL_ImgArr.astype('uint8')
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
        FL_ImgArr = np.where(baseImgarray > i, 255, 0)
        FL_ImgArr = FL_ImgArr.astype('uint8')
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
    FL_ImgArr = np.where(baseImgarray > threshholdResults[0]['threshhold'], 255, 0)
    FL_ImgArr = FL_ImgArr.astype('uint8')
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
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

# TODO Implement button recognition
# Done finish reliable text recognition
# TODO Implement tree type gui exporation. Use image similarity tests to recognize loops and determine branches

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






    buttonBuffer = pyautogui.screenshot(region=buttonWindow)
    findText(buttonBuffer)

    print('Done. Press Enter to exit')
    input()
    exit()










