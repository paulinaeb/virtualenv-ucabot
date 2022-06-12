from enum import Flag
import PySimpleGUI as sg
import cv2
import numpy as np


def main():
    sg.theme('Black')
    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1),  font='Any 14'),
               sg.Button('Exit', size=(10, 1),  font='Helvetica 14'),]]

    # create the window and show it without the plot
    window = sg.Window('Computer Vision - OpenCV', layout, element_justification='c', location=(800, 400))

    # --- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(1)
    recording = False
    
    #object detector
    object_detector = cv2.createBackgroundSubtractorMOG2()
    

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Start':
            recording = True

        elif event == 'Stop' and recording == True:
            recording = False
            img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)
            # closes the camera
            cap.release()

        if recording:
            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # define range of colors in HSV
            #mask1
            lower_blue = np.array([90,60,0])
            upper_blue = np.array([121,255,255])
            # mask2
            lower_green = np.array([40,70,80])
            upper_green = np.array([70,255,255])
            #mask3
            lower_yellow = np.array([25,70,120])
            upper_yellow = np.array([30,255,255])
            #mask4
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([350,55,100])
            
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            mask2 = cv2.inRange(hsv, lower_green, upper_green)
            mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask4 = cv2.inRange(hsv, lower_black, upper_black)
            # cv2.imshow('mask',mask) to see masks
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for count in contours: 
                epsilon = 0.01 * cv2.arcLength(count, True)
                aprox = cv2.approxPolyDP(count, epsilon, True)
                cv2.drawContours(frame, [aprox],0, (0), 3)
                i,j = aprox[0][0]
                if len(aprox) == 3:
                    cv2.putText(frame, "triangle", (i,j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            
            cv2.waitKey(1)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)



if __name__=='__main__':
    main()