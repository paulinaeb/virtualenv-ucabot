import PySimpleGUI as sg
import cv2
import numpy as np 
import math
import data

# global variables
min_prev_x = min_prev_y = max_prev_x = max_prev_y = 0
mundo = {'yellow':[],
         'green':[],
         'blue':[]
         }

# window to viewport functions
def new_x(valor, min_prev, max_prev):
    value = (((valor - min_prev) * (data.NEW_MAX - data.NEW_MIN)) / (max_prev - min_prev)) + data.NEW_MIN
    return int(value)


def new_y(valor, min_prev, max_prev):
    value = (((min_prev - valor) * (data.NEW_MAX - data.NEW_MIN)) / (min_prev - max_prev)) + data.NEW_MIN
    return int(value)

# viewport to window functions
def vp_2_w_x(value, min_prev, max_prev):
    new_value = (((value - data.NEW_MIN) * (max_prev - min_prev)) / (data.NEW_MAX - data.NEW_MIN)) + min_prev
    return int(new_value)


def vp_2_w_y(value, min_prev, max_prev):
    new_value = (((data.NEW_MIN - value) * (min_prev - max_prev)) / (data.NEW_MAX - data.NEW_MIN)) + min_prev
    return int(new_value)

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
                global min_prev_x
                min_prev_x = region[0][0]
                global min_prev_y
                min_prev_y = region[0][1]
                global max_prev_x
                max_prev_x = region[1][0]
                global max_prev_y
                max_prev_y = region[1][1] 
                # calculates and shows origin and max limit of viewport
                min_x = new_x(region[0][0], min_prev_x, max_prev_x) 
                max_x = new_x(region[1][0], min_prev_x, max_prev_x) 
                min_y = new_y(region[0][1], min_prev_y, max_prev_y)
                max_y = new_y(region[1][1], min_prev_y, max_prev_y)
                cv2.putText(frame, (str(min_x)+','+str(min_y)), (int(min_prev_x), int(min_prev_y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, (str(max_x)+','+str(max_y)), (int(max_prev_x), int(max_prev_y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                # test vp to w function
                # vx = vp_2_w_x(50, min_prev_x, max_prev_x)
                # print('x: ',str(vx))
                # vy = vp_2_w_y(50, min_prev_y, max_prev_y)
                # print('y: ',str(vy)) 
                # generating masks for other colors
                if not (generate_mask(frame, hsv_general, 'blue')):
                    mundo['blue'] = []
                if not (generate_mask(frame, hsv_general, 'yellow')):
                    mundo['yellow'] = []
                if not (generate_mask(frame, hsv_general, 'green')):
                    mundo['green'] = []
            imgbytes = cv2.imencode('.png', frame)[1].tobytes() 
            window['image'].update(data=imgbytes)
            
# function to generate each mask and draw contours and name of shapes given the color
def generate_mask(frame, hsv, color):
    mask = cv2.inRange(hsv, np.array(data.HSV_COLORS[color][0]), np.array(data.HSV_COLORS[color][1]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #variables to define the rectangle of viewport
    num_corner = 0
    first_corner = []
    second_corner = []
    
    for count in contours:  
        # using functions to get the contour of shapes
        epsilon = 0.01 * cv2.arcLength(count, True)
        approx = cv2.approxPolyDP(count, epsilon, True)
        # get area to work with only visible objects
        area = cv2.contourArea(count)
        if area > 500:
            # recognize triangles or rectangles 
            if len(approx) == 4 and color == 'black': 
                # computes the centroid of shapes
                M = cv2.moments(count)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (cx,cy), 2, (255,255,255), 2)
                # rectangles - marks
                if num_corner == 0:
                    first_corner.append(cx)
                    first_corner.append(cy) 
                    num_corner = num_corner + 1 
                elif num_corner == 1:
                    second_corner.append(cx)
                    second_corner.append(cy)
                    num_corner = num_corner + 1 
                    # draws the region of interest as a rectangle
                    cv2.rectangle(frame, (first_corner[0], first_corner[1]), (second_corner[0], second_corner[1]), (255,255,255), 2)
                    return first_corner, second_corner
                elif num_corner == 2:
                    # reset values
                    first_corner = second_corner = []
                    num_corner = 0
                        
            elif len(approx) == 3 and color !='black':
                flag = 0
                # triangles 
                x_point = []
                y_point = []
                n = approx.ravel()
                i = 0
                for j in n :
                    if(i % 2 == 0):
                        x = n[i]
                        y = n[i + 1] 
                        if (max_prev_x > x > min_prev_x) and (min_prev_y > y > max_prev_y):
                            flag = flag + 1 
                        # String containing the co-ordinates.
                        # string = str(x) + " " + str(y) 
                        # cv2.putText(frame, string, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255)) 
                        x_point.append(x)
                        y_point.append(y)
                    i = i + 1
                if flag == 3 :
                    id_agent = get_id(color)
                    cv2.drawContours(frame, [approx],0, (0), 2)
                    # computes the centroid of shapes
                    M = cv2.moments(count)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00']) 
                    # get min angle and their coordinates - min_angle - vx - vy
                    min_angle = get_angle(x_point[0], y_point[0], x_point[1], y_point[1], x_point[2], y_point[2])
                    vx = min_angle[1]
                    vy = min_angle[2]
                    direction = direction_angle(cx, cy, vx, vy)
                    cv2.putText(frame, str(id_agent), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)) 
                    global mundo
                    mundo[color] = [id_agent, cx, cy, direction]
                    # print(mundo)
                    # test rotate function
                    # if color == 'blue' and mundo['yellow']:
                    #     # 1 is left and 2 is right
                    #     rotation = rotate(direction, 20, 2)
                    #     # point where I wish to go
                    #     px = mundo['yellow'][1] 
                    #     py = mundo['yellow'][2] 
                    #     cv2.circle(frame, (px, py), 2, (255,255,255), 2)
                    #     temp = direction_angle(cx, cy, px, py)
                    #     angle_to_point = temp - direction
                    #     print('angle to point ', angle_to_point)
                    #     cv2.putText(frame,str(angle_to_point), (px, py), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    # # convert position (cx cy) to viewport 
                    # cx = new_x(cx, min_prev_x, max_prev_x) 
                    # cy = new_y(cy, min_prev_y, max_prev_y) 
                    
                    cv2.putText(frame,str(direction)+' '+str(cx)+' '+ str(cy), (min_angle[1], min_angle[2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)) 
                    return True
    return

# get if of robots in function of color given
def get_id(color):
    if color == 'blue':
        robot_id = 1
    if color == 'green':
        robot_id = 2
    if color == 'yellow':
        robot_id = 3
    return robot_id

# get direction of the triangle using min angle triangle vertices and centroid
def direction_angle(cx, cy, vx, vy):
    diff_x = vx - cx
    # cy and vy are inverted because camera orientation in Y is inverted
    diff_y = cy - vy
    # hypotenuse of the rectangle triangle formed with X and the line between C and min angle V
    h = math.sqrt((diff_x*diff_x) + (diff_y*diff_y))
    direction_angle = math.acos (diff_x / h)
    # transform result to degrees
    direction_angle = int(direction_angle * (180 / math.pi))
    if vy > cy:
        direction_angle = 360 - direction_angle 
    return direction_angle


def line_length(x1, y1, x2, y2):
    x_dif = x1-x2
    y_dif = y1-y2
    return x_dif * x_dif + y_dif * y_dif

# get all angles of given 3 points of triangle 
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
    alpha = int(alpha * 180 / math.pi);
    beta = int(beta * 180 / math.pi);
    gamma = int(gamma * 180 / math.pi);
    
    # return lower angle and its coordinates
    if gamma > alpha < beta: 
        return alpha, x1 , y1
    elif gamma > beta < alpha: 
        return beta, x2, y2
    elif beta > gamma < alpha: 
        return gamma, x3, y3


def rotate(agent, current_angle, degrees_to_rotate, direction):
    if mundo[agent]:
        result_angle = error = m1 = m2 = 0
        KP = 5
        # rotate to left
        if direction == 1 :
            result_angle = current_angle + degrees_to_rotate
        # rotate to right 
        elif direction == 2:
            result_angle = current_angle - degrees_to_rotate
        error = degrees_to_rotate
        if direction == 1:
            m1 = -error * KP    
            m2 = error * KP
        elif direction == 2:
            m1 = error * KP
            m2 = -error * KP
        print('result angle: ', result_angle)
        print('m1: ', m1,',','m2: ',m2)
        print('error: ', error)
        return result_angle, m1, m2

# verify if agents exist on viewport
def detect_agent(follow_agent, followed_agent):
    if mundo[follow_agent] and mundo[followed_agent]:
        return True
    else:
        return False

# distance between 2 points (it could be an specific point or a centroid of an agent)
def get_distance(x1, x2, y1, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print('distance:', distance)
    return distance

if __name__=='__main__':
    main()