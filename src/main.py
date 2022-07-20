from operator import length_hint
import PySimpleGUI as sg
import cv2
import numpy as np 
import math
import data
from screeninfo import get_monitors 
# from io import BytesIO 
# from PIL import Image

# world variables
agent = {'yellow': None,
         'green': None,
         'blue': None}  

class ViewPort:
    def __init__(self, name):
        self.name = name
        self.u_min = self.u_max = self.v_min = self.v_max = self.du = self.dv = None
    def set_values(self,  u_min, v_min, u_max, v_max):
        self.u_min = u_min
        self.u_max = u_max     
        self.v_min = v_min
        self.v_max = v_max
        self.du = u_max - u_min
        if self.name == 'camera':
            self.dv = v_min - v_max
        else:
            self.dv = v_max - v_min
    
class Agent:
    def __init__(self, color): 
        self.id = get_id(color)
        self.cx = self.cy = self.direction = self.line2 = self.line1 = self.limit = self.radius = None 
    def set_radius(self, r):
        self.radius = r
    def set_centroid(self, cx, cy):
        self.cx = cx
        self.cy = cy
    def set_direction(self, direction):
        self.direction = direction
    def set_line(self, line1, line2):
        self.line1 = line1
        self.line2 = line2
    def set_limit(self, limit):
        self.limit = limit
    def set_out(self):
        self.cx = self.cy = self.direction = self.line2 = self.line1 = self.limit = self.radius = None    

# window to viewport function
def w2vp(x, y, VP):
    if VP.du > 0:
        value_x = round(((x - VP.u_min) * (data.NEW_MAX_X - data.NEW_MIN_X) / VP.du) + data.NEW_MIN_X, 2)
        if y is not None:
            if VP.dv > 0:
                value_y = round(((VP.v_min - y) * (data.NEW_MAX_Y - data.NEW_MIN_Y) / VP.dv) + data.NEW_MIN_Y, 2)
                return value_x, value_y 
            else:
                return None
        return value_x
    else:
        return None

# viewport to window function
def vp2w(x, y, VP):
    value_x = round(((x - data.NEW_MIN_X) * VP.du / (data.NEW_MAX_X - data.NEW_MIN_X)) + VP.u_min, 2)
    if y is not None:
        diff_y = 0
        if VP.name == 'camera':
            diff_y = data.NEW_MIN_Y - y
        else:
            diff_y = y - data.NEW_MIN_Y
        value_y = round((diff_y * VP.dv / (data.NEW_MAX_Y - data.NEW_MIN_Y)) + VP.v_min, 2)
        return value_x, value_y
    return value_x

# viewport for projector
vpv = ViewPort('video')

# viewport for camera
vpc = ViewPort('camera')
    
# def image_to_data(im): 
#     with BytesIO() as output:
#         im.save(output, format="PNG")
#         data = output.getvalue()
#     return data
 
# draw marks and define rectangle as background
def draw_marks():
    back = draw.draw_rectangle((5, 5), ((vpv.u_max, vpv.v_max)), fill_color='black', line_color='white')
    mark1_centroid = [draw.draw_circle((5, 5), 5, fill_color='yellow')]
    mark2_centroid = [draw.draw_circle((vpv.u_max, vpv.v_max), 5, fill_color='yellow')]
    return
            
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
        if area > 10:
            # recognize triangles or rectangles 
            if len(approx) == 4 and color == 'black': 
                # computes the centroid of shapes
                M = cv2.moments(count)
                cx = round(M['m10'] / M['m00'], 2)
                cy = round(M['m01'] / M['m00'], 2)
                cv2.circle(frame, (int(cx),int(cy)), 2, (255,255,255), 2)
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
                    cv2.rectangle(frame, (int(first_corner[0]), int(first_corner[1])), (int(second_corner[0]), int(second_corner[1])), (255,255,255), 2)
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
                        if (vpc.u_max > x > vpc.u_min) and (vpc.v_min > y > vpc.v_max):
                            flag = flag + 1 
                        # String containing the co-ordinates.
                        # string = str(x) + " " + str(y) 
                        # cv2.putText(frame, string, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255)) 
                        x_point.append(x)
                        y_point.append(y)
                    i = i + 1
                if flag == 3 : 
                    cv2.drawContours(frame, [approx],0, (0), 2)
                    # computes the centroid of shapes
                    M = cv2.moments(count)
                    cx = round(M['m10'] / M['m00'], 2)
                    cy = round(M['m01'] / M['m00'], 2) 
                    cv2.circle(frame, (int(cx), int(cy)), 1, (0, 0, 0))
                    new_agent = Agent(color)
                    new_agent.set_centroid(cx, cy)
                    # get min angle and their coordinates - min_angle - vx - vy
                    min_angle, vx, vy = get_angle(x_point[0], y_point[0], x_point[1], y_point[1], x_point[2], y_point[2])
                    # get direction of min angle (vertex) represents agent's direction
                    direction = direction_angle(cx, cy, vx, vy)
                    
                    print('angle', direction)
                    print('first cx', cx, 'cy', cy)
                    new_agent.set_direction(direction)
                    cv2.putText(frame, str(new_agent.id), (int(cx), int(cy)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)) 
                    # distance between centroid and min angle vertex  
                    length= distance(new_agent, vx, vy) 
                    print('distance', length)
                    # radius of agent limits 
                    r = round(length + length * .12, 2)
                    new_agent.set_radius(r) 
                    length = w2vp(length, None, vpc)
                    print('length', length)
                    length = -1 * length
                    p1 = length + (length / 3)
                    p2 = p1 + length
                    p3 = -1 * p1
                    p4 = -1 * p2
                    # create agent in the world
                    global agent
                    # agent[color] = [id_agent, cx, cy, direction] 
                    # convert position (cx cy) to viewport 
                    agent[color] = new_agent
                    new_agent.set_limit(draw_treshold(frame, r, new_agent, 'green'))
                    for a in agent.values():
                        if a is not None:
                            if a.id is not new_agent.id:
                                if a.cx is not None: 
                                    d = get_distance(a.cx, a.cy, new_agent.cx, new_agent.cy)
                                    r_sum = a.radius + r
                                    # print('a', new_agent.id,'b',a.id, 'dis', d, 'r sum', r_sum) 
                                    if d < r_sum: 
                                        new_agent.set_limit(draw_treshold(frame, r, new_agent, 'red'))
                                    else:
                                        new_agent.set_limit(draw_treshold(frame, r, new_agent, 'green'))
                                        
                    cx, cy = w2vp(cx, cy, vpc) 
                    # calculate matrix transformation-rotation
                    MT = get_MT(cx, cy, direction) 
                    # translate points and draw line
                    new_agent.set_line(translate_point(MT, frame, p1, 0, p2, 0), translate_point(MT, frame, p3, 0, p4, 0))
                    print('cx', cx, 'cy', cy)
                    cv2.putText(frame,str(direction)+' '+str(cx)+' '+ str(cy), (vx, vy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0)) 
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
    h = math.sqrt((diff_x * diff_x) + (diff_y * diff_y))
    direction_angle = math.acos (diff_x / h)
    # transform result to degrees
    direction_angle = round(direction_angle * 180 / math.pi, 2)
    if vy > cy:
        direction_angle = 360 - direction_angle 
    return direction_angle


def line_length(x1, y1, x2, y2):
    x_dif = x1 - x2
    y_dif = y1 - y2
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
    alpha = round(alpha * 180 / math.pi, 2);
    beta = round(beta * 180 / math.pi, 2);
    gamma = round(gamma * 180 / math.pi, 2);
    
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
    distance = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
    return distance

# distance from centroid to min angle vertex
def distance(agent, vx, vy):
    value = round(math.sqrt((agent.cx - vx)**2 + (agent.cy - vy)**2), 2)
    return value


def translate_point(MT, frame, px1, py1, px2, py2):
    P1 = np.array([[px1], [py1], [1]]) 
    P1P = MT.dot(P1)
    P2 = np.array([[px2], [py2], [1]]) 
    P2P = MT.dot(P2)
    # transform vp 2 camera
    p1p_xc, p1p_yc = vp2w(int(P1P[0]), int(P1P[1]), vpc)
    p2p_xc, p2p_yc = vp2w(int(P2P[0]), int(P2P[1]), vpc)
    # transform vp 2 vb
    p1p_xv, p1p_yv = vp2w(int(P1P[0]), int(P1P[1]), vpv)
    p2p_xv, p2p_yv = vp2w(int(P2P[0]), int(P2P[1]), vpv) 
    # draw lines
    cv2.line(frame, (int(p1p_xc), int(p1p_yc)), (int(p2p_xc), int(p2p_yc)), (255, 0, 0), 2) 
    line = [draw.draw_line((p1p_xv, p1p_yv), (p2p_xv, p2p_yv), color='blue')] 
    return line


def get_MT(cx, cy, alpha):
    gamma =  alpha * np.pi / 180 
    # rotation
    R = np.array([[np.cos(gamma), - np.sin(gamma)],
                [np.sin(gamma), np.cos(gamma)],
                [0, 0]])   
    column = np.array([[0], [0], [1]])
    # final rotation
    R = np.concatenate((R, column), axis = 1) 
    # translation
    P = np.array([[cx], [cy], [1]]) 
    T = np.array([[1, 0], [0, 1], [0,0]])
    T = np.concatenate((T, P), axis = 1) 
    # translation * rotation matrix
    MT = T.dot(R) 
    return MT

def draw_treshold(frame, r, ob, color): 
    # get centroid in vpv
    if color == 'green':
        cv2.circle(frame, (int(ob.cx), int(ob.cy)), int(r), (0, 255, 0), 2)
    else:
        cv2.circle(frame, (int(ob.cx), int(ob.cy)), int(r), (0, 0, 255), 2)
    cxc, cyc = w2vp(ob.cx, ob.cy, vpc)
    cxv, cyv = vp2w(cxc, cyc, vpv)
    # get radius in vpv
    rv = w2vp(r, None, vpc)
    rvb = vp2w(rv, None, vpv) 
    circle = [draw.draw_circle((cxv, cyv), rvb, line_color=color)]
    return circle


def detect_object(a, b):
    
    pass

# generates masks for all agent colors and clears projection when needed
def clear_figures(frame, hsv_general):
    for color in agent.keys():
        if (not generate_mask(frame, hsv_general, color)) and (agent[color] is not None):
            if not one_monitor:
                draw.delete_figure(agent[color].line1)
                draw.delete_figure(agent[color].line2)
                draw.delete_figure(agent[color].limit)
            agent[color].set_out()
    return


# it's needed to activate virtual environment before running this
def main():
    sg.theme('Black')

    # define the window layout
    layout = [[sg.Text('Virtual Environment', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Start', size=(10, 1), font='Helvetica 14'), 
               sg.Button('Exit', size=(10, 1),  font='Helvetica 14')]] 

    # create the window and show it without the plot
    window = sg.Window('Virtual Environment', layout, element_justification='c', location=(350, 100))
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
            x_init = 0
            recording = True
            # generate projection
            for m in get_monitors(): 
                global one_monitor
                one_monitor = m.is_primary
                x_init = m.x
                global vpv
                 # set viewport values for projection 
                vpv.set_values(5, 5, m.width - 5, m.height - 5) 
            # im_mark = Image.open('../img/equis.png')
            # im_mark_new = im_mark.resize((mark_size, mark_size)) 
            layout = [[sg.Graph((vpv.u_max + 5, vpv.v_max + 5), (0, 0), (vpv.u_max + 5, vpv.v_max + 5), enable_events=True, key='-GRAPH-', pad=(0,0))]]
            virtual_window = sg.Window('Virtual world', layout, no_titlebar=True, finalize=True, location=(x_init,0), size=(vpv.u_max + 5, vpv.v_max + 5), margins=(0,0)).Finalize()
            virtual_window.Maximize() 
            global draw
            draw = virtual_window['-GRAPH-']  
            
        if recording: 
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
                # print('region',region) 
                # calculates and shows origin and max limit of viewport 
                global vpc 
                vpc.set_values(region[0][0], region[0][1], region[1][0], region[1][1]) 
                # convert limits coordinates to vp
                vpc_min  = w2vp(region[0][0], region[0][1], vpc) 
                vpc_max  = w2vp(region[1][0], region[1][1], vpc)  
                if (vpc_min and vpc_max):
                    cv2.putText(frame, (str(int(vpc_min[0]))+','+str(int(vpc_min[1]))), (int(vpc.u_min) - 10, int(vpc.v_min) + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, (str(int(vpc_max[0]))+','+str(int(vpc_max[1]))), (int(vpc.u_max) - 70, int(vpc.v_max) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                    # generating masks for all agent colors and clearing screen (vb) when needed
                    clear_figures(frame, hsv_general) 
                    # blue follows yellow just for testing
                    # detect_and_follow_agent(frame, 'blue', 'yellow') 
                    # ids = [draw.draw_image('../img/eggs.png', location=(x_dog, y_dog))] 
            elif (not region) and (not one_monitor): 
                draw.delete_figure('all')
                draw_marks()  
            imgbytes = cv2.imencode('.png', frame)[1].tobytes() 
            window['image'].update(data=imgbytes)


if __name__=='__main__':
    main()
    
    
