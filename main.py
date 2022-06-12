from enum import Flag
import PySimpleGUI as sg
import cv2
import numpy as np

# constants
# colors in hsv dict for masks. the first value represents the lower limit and the second the lower
HSV_COLORS = {
    'blue': [[90,60,0], [121,255,255]], 
    'green': [[40,70,80], [70,255,255]],
    'yellow': [[20, 100, 100], [30, 255, 255]],
    'black': [[0, 0, 0], [180,255,30]]
}

# remember to activate virtual environment before running this
def main():
    sg.theme('Black')
    # define the window layout
    layout = [[sg.Text('Virtual Environment', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1),  font='Any 14'),
               sg.Button('Exit', size=(10, 1),  font='Helvetica 14'),]]

    # create the window and show it without the plot
    window = sg.Window('Virtual Environment', layout, element_justification='c', location=(350, 150))
    #indicates which camera use
    cap = cv2.VideoCapture(0)
    recording = False

    # Event LOOP Read and display frames, operate the GUI 
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            cap.release()
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
            mask = cv2.inRange(hsv, np.array(HSV_COLORS['black'][0]), np.array(HSV_COLORS['black'][1]))
            # cv2.imshow('mask',mask) to see masks
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for count in contours: 
                epsilon = 0.01 * cv2.arcLength(count, True)
                aprox = cv2.approxPolyDP(count, epsilon, True)
                cv2.drawContours(frame, [aprox],0, (0), 3)
                i,j = aprox[0][0]
                if len(aprox) == 3:
                    cv2.putText(frame, "triangle", (i,j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
                elif len(aprox) == 4:
                    cv2.putText(frame, "rectangle", (i,j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            
            cv2.waitKey(1)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)


def generate_mask():
    pass


if __name__=='__main__':
    main()