from io import BytesIO
from random import randint
from PIL import Image
import PySimpleGUI as sg 

width, height = size = (1000, 450)

def image_to_data(im): 
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

im_mark = Image.open('../img/equis.png')
im_mark_new = im_mark.resize((25,25))

layout = [[sg.Graph(size, (0, 0), size, enable_events=True, key='-GRAPH-', pad=(0,0))]]
virtual_window = sg.Window('Virtual world', layout, no_titlebar=True, finalize=True, location=(0,0), size=(width,height), margins=(0,0)).Finalize()
virtual_window.Maximize()
draw = virtual_window['-GRAPH-']

# background = draw.draw_image(data=image_to_data(new_im), location=(0, height))
back = draw.draw_rectangle((0, height), (width, 0), fill_color='black')
# 25 is the error margin for the window
ids = [draw.draw_image('../img/eggs.png', location=(randint(0, width-25), randint(0, height-25)))]
mark1 = [draw.draw_image(data=image_to_data(im_mark_new), location=(0, 25))]
mark2 = [draw.draw_image(data=image_to_data(im_mark_new), location=(width-25, height))]
dog = [draw.draw_circle((75, 75), 25, fill_color='white', line_color='white')]

while True:
    # c = cv2.wind
    event2, values2 = virtual_window.read() 
    if event2 in (sg.WINDOW_CLOSED, 'Exit'):
        break
    # elif event == '-GRAPH-':
    #     location = values[event]
    #     figures = draw.get_figures_at_location(location)
    #     cv2.circle(values,(100,100), 63, (0,0,255), 2)
    #     for figure in figures:
    #         if figure != background:
    #             draw.delete_figure(figure)
    #             ids.remove(figure)
    # to delete img
    #     for figure in ids:
    #         draw.delete_figure(figure)

virtual_window.close()