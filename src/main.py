import PySimpleGUI as sg
import cv2
import numpy as np 
import math
import data
from screeninfo import get_monitors
from io import BytesIO
from random import randint
from PIL import Image

# world variables
agent = {'yellow':[],
         'green':[],
         'blue':[] 
         }

display = [] 

draw = 0

# viewport for projector
vpv = []

# viewport for camera
vpc = []

class ViewPort:
    def __init__(self, name, u_min, v_min, u_max, v_max):
        self.name = name
        self.u_min = u_min
        self.u_max = u_max     
        self.v_min = v_min
        self.v_max = v_max
        self.du = u_max - u_min
        if name == 'camera':
            self.dv = v_min - v_max
        else:
            self.dv = v_max - v_min     
    
    
def w2vp(x, y, VP):
    value_x = ((x - VP.u_min) * (data.NEW_MAX_X - data.NEW_MIN_X) / VP.du) + data.NEW_MIN_X
    value_y = ((VP.v_min - y) * (data.NEW_MAX_Y - data.NEW_MIN_Y) / VP.dv) + data.NEW_MIN_Y
    return round(value_x), round(value_y)


class Window:
    def __init__(self, w_min, w_max):
        self.w_min = w_min
        self.w_max = w_max
        self.w_dx = w_max.x - w_min.x
        self.w_dy = w_max.y - w_min.y


# viewport to window functions
def vp_2_w_x(value, min_prev, max_prev):
    new_value = (((value - data.NEW_MIN_X) * (max_prev - min_prev)) / (data.NEW_MAX_X - data.NEW_MIN_X)) + min_prev
    return round(new_value)


def vp_2_w_y(value, min_prev, max_prev):
    new_value = (((data.NEW_MIN_Y - value) * (min_prev - max_prev)) / (data.NEW_MAX_Y - data.NEW_MIN_Y)) + min_prev
    return round(new_value)


def vpv_2_w_y(value, min_prev, max_prev):
    new_value = (((value - data.NEW_MIN_Y) * (max_prev - min_prev)) / (data.NEW_MAX_Y - data.NEW_MIN_Y)) + min_prev
    return round(new_value)


def image_to_data(im): 
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data


# remember to activate virtual environment before running this
def main():
    sg.theme('Black')

    # define the window layout
    layout = [[sg.Text('Virtual Environment', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1),  font='Any 14'),
               sg.Button('Exit', size=(10, 1),  font='Helvetica 14')]]


    # create the window and show it without the plot
    window = sg.Window('Virtual Environment', layout, element_justification='c', location=(350, 150))
    #indicates which camera use
    cap = cv2.VideoCapture(0)
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
            # generate projection
            for m in get_monitors():
                global display
                if m.is_primary is False: 
                    display = [(m.x), (m.y), (m.width), (m.height)]
                elif m.is_primary is True: 
                    display = [(m.x), (m.y), (m.width), (m.height)]
                     
            # create viewport for video beam projection
            global vpv
            # 25 is the error margin for the window
            vpv = ViewPort('video', 0, 25, display[2]-25, display[3])
            
            # draw marks and define rectangle as background
            im_mark = Image.open('../img/equis.png')
            im_mark_new = im_mark.resize((25,25))
            layout = [[sg.Graph(((vpv.u_max + 25, vpv.v_max)), (0, 0), (vpv.u_max + 25, vpv.v_max), enable_events=True, key='-GRAPH-', pad=(0,0))]]
            virtual_window = sg.Window('Virtual world', layout, no_titlebar=True, finalize=True, location=(display[0],0), size=(vpv.u_max + 25, vpv.v_max), margins=(0,0)).Finalize()
            virtual_window.Maximize()
            global draw
            draw = virtual_window['-GRAPH-']
            back = draw.draw_rectangle((0, vpv.v_max), (vpv.u_max + 25, 0), fill_color='black')
            # ids = [draw.draw_image('../img/eggs.png', location=(randint(0, display[2]-25), randint(0, display[3]-25)))]
            mark1 = [draw.draw_image(data=image_to_data(im_mark_new), location=(vpv.u_min, vpv.v_min))]
            mark2 = [draw.draw_image(data=image_to_data(im_mark_new), location=(vpv.u_max, vpv.v_max))]  

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
            # event2, values2 = virtual_window.read() 
            # if event2 in (sg.WINDOW_CLOSED, 'Exit'):
            #     break
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
                # calculates and shows origin and max limit of viewport 
                global vpc
                vpc = ViewPort('camera', region[0][0], region[0][1], region[1][0], region[1][1])
                vpc_min = (w2vp(region[0][0], region[0][1], vpc))
                vpc_max = (w2vp(region[1][0], region[1][1], vpc))  
                # print (region)
                cv2.putText(frame, (str(vpc_min[0])+','+str(vpc_min[1])), (vpc.u_min, vpc.v_min), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, (str(vpc_max[0])+','+str(vpc_max[1])), (vpc.u_max, vpc.v_max), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                # test vp to w function
                vx = vp_2_w_x(65, vpc.u_min, vpc.u_max) 
                vy = vp_2_w_y(50, vpc.v_min, vpc.v_max) 
                # print('vx, vy ', vx,' ',vy)
                cv2.circle(frame, (vx, vy), 10, (255,0,255), 2)
                # generating masks for other colors
                if not (generate_mask(frame, hsv_general, 'blue')):
                    agent['blue'] = []
                if not (generate_mask(frame, hsv_general, 'yellow')):
                    agent['yellow'] = []
                if not (generate_mask(frame, hsv_general, 'green')):
                    agent['green'] = []
                # blue follows yellow just for testing
                # detect_and_follow_agent(frame, 'blue', 'yellow')
                # window to vp dog
                x_dog = vp_2_w_x(65, vpv.u_min, vpv.u_max)
                y_dog = vpv_2_w_y(50, vpv.v_min, vpv.v_max) 
                dog = [draw.draw_circle((x_dog, y_dog), 20, fill_color='white', line_color='white')]
                # ids = [draw.draw_image('../img/eggs.png', location=(x_dog, y_dog))]
            imgbytes = cv2.imencode('.png', frame)[1].tobytes() 
            window['image'].update(data=imgbytes)
    virtual_window.close()

            
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
                        # this verifies that every vertex is in the region of the viewport
                        # print(vpc.u_min)
                        if (vpc.u_max > x > vpc.u_min) and (vpc.v_min > y > vpc.v_max):
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
                    global agent
                    agent[color] = [id_agent, cx, cy, direction]
                    # print(agent)
                    # test rotate function 
                    #     # 1 is left and 2 is right
                    #     rotation = rotate(direction, 20, 2)
                    # convert position (cx cy) to viewport 
                    cx, cy = w2vp(cx, cy, vpc) 
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
    if agent[agent]:
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
def detect_and_follow_agent(frame, follow_agent, followed_agent):
    if agent[follow_agent] and agent[followed_agent]:
        # point where I wish to go
        cv2.line(frame, (agent[follow_agent][1], agent[follow_agent][2]), (agent[followed_agent][1], agent[followed_agent][2]), (255,0,0), 2)
        temp = direction_angle(agent[follow_agent][1], agent[follow_agent][2], agent[followed_agent][1], agent[followed_agent][2])
        angle_to_point = temp - agent[follow_agent][3]
        print('angle to point ', angle_to_point)
        cv2.putText(frame,str(angle_to_point), (agent[followed_agent][1], agent[followed_agent][2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
    else:
        return False

# distance between 2 points (it could be an specific point or a centroid of an agent)
def get_distance(x1, x2, y1, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print('distance:', distance)
    return distance

if __name__=='__main__':
    main()