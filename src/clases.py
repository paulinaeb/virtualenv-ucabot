import data

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
            
            
# value = (((valor - min_prev) * (data.NEW_MAX_X - data.NEW_MIN_X)) / (max_prev - min_prev)) + data.NEW_MIN_X
# value = (((min_prev - valor) * (data.NEW_MAX_Y - data.NEW_MIN_Y)) / (min_prev - max_prev)) + data.NEW_MIN_Y
            
region = []

region = ViewPort('camera', 2, 2, 5, 1)
print(region.name, ' ', region.du,' ', region.dv, ' ', region.u_max)

print(w2vp(10, 9, region))