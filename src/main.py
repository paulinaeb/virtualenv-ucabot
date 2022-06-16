from cmath import acos
import PySimpleGUI as sg
import cv2
import numpy as np 
import math
import data


def new_value(valor, min_prev, max_prev):
    value = (((valor - min_prev) / (max_prev - min_prev))*(data.NEW_MAX - data.NEW_MIN) + data.NEW_MIN)
    return value

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
    cap = cv2.VideoCapture(1)
    recording = False
    # Event loop Read and display frames, operate the GUI 
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            if recording:
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
            frame = 0
            _, frame = cap.read()
            # converting image obtained to hsv, if exists
            if frame is None:
                print('Something went wrong trying to connect to your camera. Please verify.')
                return
            else:
                hsv_general = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # generate mask to define region of interest (viewport)
            region = generate_mask(frame, hsv_general, 'black')
            # if two black marks exist 
            if region:  
                # define region of interest and viewport
                if region[1][1] > region[0][1] and region[0][0] > region[1][0]:
                    roi = frame[region[0][1]:region[1][1], region[1][0]:region[0][0]]
                    
                elif region[1][1] > region[0][1] and region[0][0] < region[1][0]:
                    roi = frame[region[0][1]:region[1][1],region[0][0]:region[1][0]]
               
                elif region[1][1] < region[0][1] and region[0][0] < region[1][0]:
                    roi = frame[region[1][1]:region[0][1], region[0][0]:region[1][0]]
                  
                elif region[1][1] < region[0][1] and region[0][0] > region[1][0]:
                    roi = frame[region[1][1]:region[0][1], region[1][0]:region[0][0]]
                print(region)
                # getting hsv of viewport
                hsv_region = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # generating masks for other colors
                generate_mask(roi, hsv_region, 'blue')
                generate_mask(roi, hsv_region, 'yellow')
                generate_mask(roi, hsv_region, 'green') 
                # shows origin and max limit of viewport
                cv2.putText(frame, (str(data.NEW_MIN)+','+str(data.NEW_MIN)), (int(region[1][0]), int(region[1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                cv2.putText(frame, (str(data.NEW_MAX)+','+str(data.NEW_MAX)), (int(region[0][0]), int(region[0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
            imgbytes = cv2.imencode('.png', frame)[1].tobytes() 
            window['image'].update(data=imgbytes)
            

# function to generate each mask and draw contours and name of shapes given the color
def generate_mask(frame, hsv, color):
    mask = cv2.inRange(hsv, np.array(data.HSV_COLORS[color][0]), np.array(data.HSV_COLORS[color][1]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #variables to define the area
    num_corner = 0
    first_corner = []
    second_corner = []
    # get area, and show name of shape
    for count in contours:  
        epsilon = 0.01 * cv2.arcLength(count, True)
        approx = cv2.approxPolyDP(count, epsilon, True)
        area = cv2.contourArea(count)
        if area > 500:
            if len(approx) == 3 or (len(approx) == 4 and color == 'black'):
                cv2.drawContours(frame, [approx],0, (0), 3)
                i,j = approx[0][0]
                # computes the centroid of shapes
                M = cv2.moments(count)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (cx,cy), 3, (255,255,255), 2)
                if len(approx) == 4 and color == 'black': 
                    # rectangles - marks
                    if num_corner == 0:
                        first_corner.append(cx)
                        first_corner.append(cy) 
                        num_corner = num_corner + 1 
                    elif num_corner == 1:
                        second_corner.append(cx)
                        second_corner.append(cy)
                        num_corner = num_corner + 1 
                        cv2.rectangle(frame, (first_corner[0], first_corner[1]), (second_corner[0], second_corner[1]), (255,255,255), 2)
                        return first_corner, second_corner
                    elif num_corner == 2:
                        first_corner = second_corner = []
                        num_corner = 0
                elif len(approx) == 3 and color !='black':
                    diff_x = diff_y = direction_angle = 0
                    x_point = []
                    y_point = []
                    # triangles
                    n = approx.ravel()
                    i = 0
                    for j in n :
                        if(i % 2 == 0):
                            x = n[i]
                            y = n[i + 1]
                            # String containing the co-ordinates.
                            # string = str(x) + " " + str(y) 
                            # cv2.putText(frame, string, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0)) 
                            x_point.append(x)
                            y_point.append(y)
                        i = i + 1
                    min_angle = get_angle(x_point[0], y_point[0], x_point[1], y_point[1], x_point[2], y_point[2])
                    # # get direction of the triangle using min angle triangle vertices and centroid
                    if cx and cy and min_angle[1] and min_angle[0]:
                        diff_x = min_angle[1] - cx
                        diff_y = min_angle[2] - cy
                        direction_angle = int(math.atan2(diff_y, diff_x) * (180 / math.pi))
                        print('Angulo en grados: ' + str(direction_angle))
                        cv2.putText(frame, str(direction_angle), (min_angle[1], min_angle[2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0)) 
                    # print('min angle: ', min_angle[0], 'vertice:', min_angle[1],' ', min_angle[2])
                    # cv2.putText(frame, str(int(min_angle[0])), (min_angle[1], min_angle[2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0)) 
    return


def line_length(x1, y1, x2, y2):
    x_dif = x1-y1
    y_dif = x2-y2
    return x_dif * x_dif + y_dif * y_dif


def get_angle(x1, y1, x2, y2, x3, y3):
    # bc
    a2 = line_length(x2, y2, x3, y3)
    # ac
    b2 = line_length(x1, y1, x3, y3)
    # ab
    c2 = line_length(x1, y1, x2, y2)
    
    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)
    
    alpha = math.acos((b2 + c2 - a2) / (2 * b * c))
    beta = math.acos((a2 + c2 -b2) / (2 * a * c))
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b))
    # Converting to degree
    alpha = alpha * 180 / math.pi;
    beta = beta * 180 / math.pi;
    gamma = gamma * 180 / math.pi;
    # return lower angle
    if gamma > alpha < beta:
        return alpha, x1 , y1
    elif gamma > beta < alpha:
        return beta, x3, y3
    elif beta > gamma < alpha:
        return gamma, x2, y2


if __name__=='__main__':
    main()