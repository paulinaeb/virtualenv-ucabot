import data

class ViewPort:
    def __init__(self, name):
        self.name = name
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
            
region = ViewPort('camera')
region.set_values(2, 2, 5, 5)
print(region.name,' ',region.u_min,' ',region.u_max,' ', region.du,' ', region.dv)
         

# def w2vp(x, y, VP):
#     value_x = ((x - VP.u_min) * (data.NEW_MAX_X - data.NEW_MIN_X) / VP.du) + data.NEW_MIN_X
#     value_y = ((VP.v_min - y) * (data.NEW_MAX_Y - data.NEW_MIN_Y) / VP.dv) + data.NEW_MIN_Y
#     return round(value_x), round(value_y)
            
            
# # value = (((valor - min_prev) * (data.NEW_MAX_X - data.NEW_MIN_X)) / (max_prev - min_prev)) + data.NEW_MIN_X
# # value = (((min_prev - valor) * (data.NEW_MAX_Y - data.NEW_MIN_Y)) / (min_prev - max_prev)) + data.NEW_MIN_Y
            
# region = []

# region = ViewPort('camera', 2, 2, 5, 1)
# print(region.name, ' ', region.du,' ', region.dv, ' ', region.u_max)

# print(w2vp(10, 9, region))

# class Window:
#     def __init__(self, name, x_min, y_min, x_max, y_max):
#         self.name = name
#         self.x_min = x_min
#         self.y_min = y_min
#         self.x_max = x_max
#         self.y_max = y_max
#         self.dx = x_max - x_min
#         if name == 'camera':
#             self.dy = y_min - y_max
#         else:
#             self.dy = y_max - y_min 
            
            
# region = Window('camera', 4, 4, 10, 10)

# region.name = 'video'
# print(region.name)

# def vp2w(x, y, W):
#     value_x = ((x - data.NEW_MIN_X) * W.dx / (data.NEW_MAX_X - data.NEW_MIN_X)) + W.x_min
#     diff_y = 0
#     if W.name == 'camera':
#         diff_y = data.NEW_MIN_Y - y
#     else:
#         diff_y = y - data.NEW_MIN_Y
#     value_y = (diff_y * W.dy / (data.NEW_MAX_Y - data.NEW_MIN_Y)) + W.y_min
#     return round(value_x), round(value_y)


         
# def vp_2_w_x(value, min_prev, max_prev):
#     new_value = (((value - data.NEW_MIN_X) * (max_prev - min_prev)) / (data.NEW_MAX_X - data.NEW_MIN_X)) + min_prev
#     return round(new_value)


# def vp_2_w_y(value, min_prev, max_prev):
#     new_value = (((data.NEW_MIN_Y - value) * (min_prev - max_prev)) / (data.NEW_MAX_Y - data.NEW_MIN_Y)) + min_prev
#     return round(new_value)


# def vpv_2_w_y(value, min_prev, max_prev):
#     new_value = (((value - data.NEW_MIN_Y) * (max_prev - min_prev)) / (data.NEW_MAX_Y - data.NEW_MIN_Y)) + min_prev
#     return round(new_value)