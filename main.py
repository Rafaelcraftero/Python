import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.config import Config
from kivy.graphics import Color, Line, InstructionGroup
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image as kiImage

import kivy.clock
import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from skimage.segmentation import random_walker
from skimage.filters import edges

from io import BytesIO
from PIL import Image, ImageTk, ImageOps

Config.set('graphics', 'window_state', 'hidden')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.write()
windWidth = 550
windHeight = 450
img = Image.Image
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
color = RED
file = ""
onEditMode=False
size = 1
objects=[]

def open_file():
    global file
    global onEditMode
    canvas_x_padding = 100
    canvas_y_padding = 100
    #Busca la ruta de la imagen que vamos a cargar
    file = filedialog.askopenfilename(initialdir='/', title='Seleccionar imagen',
                                      filetypes=(('PNG Files', '.png'), ('JPG Files', '.jpg; .jpeg')))
    if file:
        global img
        onEditMode = False
        #Guarda la imagen en memoria
        img = Image.open(file)
        kk = img
        if img.width > (windWidth - canvas_x_padding):
            new_width = windWidth - canvas_x_padding
            kk = img.resize((new_width, int(img.height * new_width / img.width)))
            #Ajusta el alto a la celda del panel principal

        elif img.height > (windHeight - canvas_y_padding):
            new_height = windHeight - canvas_y_padding
            kk = kk.resize((int(kk.width * new_height / kk.height), new_height))
            #Ajusta el ancho a la celda del panel principal

        #Convierte la imagen a la libreria Tkinter y lo cambia en la interfaz
        tkimg = ImageTk.PhotoImage(kk)
        canvas.config(width=img.width, height=img.height)
        canvas.itemconfig(container, image=tkimg)
        #Si está abierto la ventana del editor se cierra
        Window.close()

    canvas.mainloop()


class PaintWindow(Widget):
    global color
    drawing = False

    def on_touch_up(self, touch):
        #Cuando levantas el click deja de dibujar
        self.drawing = False

    def on_touch_move(self, touch):
        global objects
        #Si el cursor coincide con la imagen puede dibujar ** Hay que ponerlo ya que la aplicación podría seguir dibugando en toda la ventana principal si no se añade
        if self.parent.children[0].collide_point(*touch.pos):
            try:
                if self.drawing:
                    #Añade al objeto los puntos que sigue mintras no levante el click
                    self.points.append(touch.pos)
                    self.obj.children[-1].points = self.points
                else:
                    self.drawing = True
                    #Declara un objeto y le añade la posición donde empieza a dibujar en un objeto, le añade color y los alinea con el tamaño correspondiente
                    self.points = [touch.pos]
                    self.obj = InstructionGroup()
                    self.obj.add(Color(rgb=(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)))
                    self.obj.add(Line(points=self.points, width=size))
                    #Guarda el objeto en la lista y lo añade al canvas
                    objects.append(self.obj)
                    self.canvas.add(self.obj)
            except:
                pass
        return super(PaintWindow, self).on_touch_move(touch)

    def undo_draw(self, touch):
        global objects
        #Si la lista contiene al menos 1 elemento coge el último de la lista, lo eliminas de la lista y lo eliminas del canvas
        if len(objects)>0:
            item = objects.pop(-1)
            self.canvas.remove(item)


def reset():
    #Reinicia los valores de kivy
    import kivy.core.window as window
    from kivy.base import EventLoop
    if not EventLoop.event_listeners:
        from kivy.cache import Cache
        window.Window = window.core_select_lib('window', window.window_impl, True)
        Cache.print_usage()
        for cat in Cache._categories:
            Cache._objects[cat] = {}


def start_edit():
    global file
    #Si la imagen está cargada reinicia la el editor y lanza la app
    if file:
        reset()
        Window.show()
        app=PaintApp()
        app.run()


class PaintApp(App):

    def build(self):

        global img
        global file
        global size
        self.current_i = 0
        global onEditMode
        #Ajusta la ventana al tamaño de la imagen
        if img.width > Window.width:
            Window.size = (img.width, Window.height)
        if img.height > Window.height:
            Window.size = (Window.width, img.height)

        #Crea el contenedor principal
        self.rootWindow = RelativeLayout(size=Window.size)

        #Crea el canvas donde se pintará
        self.painter = PaintWindow()
        self.PaintWindow = RelativeLayout(size_hint_y=None, height=img.height, size_hint_x=None, width=img.width,
                                     pos_hint={'center_x': 0.5, 'center_y': .5})


        beeld = kiImage()  # only use this line in first code instance
        #Si la imagen esta en edición no la pases a gris

        data = BytesIO()
        if not onEditMode:
            img = ImageOps.grayscale(img)
        #Convierte la imagen del PIL IMAGE a Kivy Image
        img.save(data, format='png')
        data.seek(0)  # yes you actually need this
        im = CoreImage(BytesIO(data.read()), ext='png')

        beeld.texture = im.texture
        #Mantiene la forma de la imagen en la ventana
        beeld.keep_ratio=True
        beeld.allow_stretch=True
        #Unes el paint con la imagen en el mismo contenedor que luego añadiremos a la ventana principal
        self.PaintWindow.add_widget(beeld)
        self.PaintWindow.add_widget(self.painter)

        #Crea los botones y los posicionas
        self.AcceptBtn = Button(text='Accept', size_hint=(0.15, 0.15), font_size='20sp',
                                pos_hint={'center_x': 0.1, 'center_y': .9}, background_color=(1, 1, 1, 0.3), color=(1,1,0,1))
        self.CancelBtn = Button(text='Cancel', size_hint=(0.15, 0.15), font_size='20sp',
                                pos_hint={'center_x': 0.85, 'center_y': .9}, background_color=(1, 1, 1, 0.3), color=(1,1,0,1))
        self.clearnBtn = Button(text='Clear', size_hint=(0.15, 0.15), font_size='20sp',
                                pos_hint={'center_x': 0.1, 'center_y': .1}, background_color=(1, 1, 1, 0.3), color=(1,1,0,1))
        self.ButtonRED = Button(text='RED', size_hint=(0.15, 0.15), font_size='20sp',
                                pos_hint={'center_x': .325, 'center_y': .1}, background_color=(1, 1, 1, 0.3), color=(1,0,0,1))
        self.ButtonGREEN = Button(text='GREEN', size_hint=(0.15, 0.15), font_size='20sp',
                                  pos_hint={'center_x': .475, 'center_y': .1}, background_color=(1, 1, 1, 0.3), color=(0,1,0,1))
        self.ButtonBLUE = Button(text='BLUE', size_hint=(0.15, 0.15), font_size='20sp',
                                 pos_hint={'center_x': .625, 'center_y': .1}, background_color=(1, 1, 1, 0.3), color=(0,0,1,1))
        self.ButtonUNDO = Button(text='UNDO', size_hint=(0.15, 0.15), font_size='20sp',
                                   pos_hint={'center_x': .85, 'center_y': .1}, background_color=(1, 1, 1, 0.3), color=(1,1,0,1))
        self.ButtonIncreaseSize = Button(text='+', size_hint=(0.10, 0.10), font_size='40sp',
                                         pos_hint={'center_x': .85, 'center_y': .6}, background_color=(1, 1, 1, 0.3), color=(1,1,0,1))
        self.ButtonDecreaseSize = Button(text='-', size_hint=(0.10, 0.10), font_size='40sp',
                                         pos_hint={'center_x': .85, 'center_y': .4}, background_color=(1, 1, 1, 0.3), color=(1,1,0,1))
        self.Size = Button(text=str(size), size_hint=(0.10, 0.10), font_size='40sp',
                           pos_hint={'center_x': .85, 'center_y': .5}, background_color=(1, 1, 1, 0), color=(1,1,0,1))
        #Inicializas con el boton rojo
        self.ButtonRED.state = "down"
        self.ButtonBLUE.state = "normal"
        self.ButtonGREEN.state = "normal"

        #Añadeir las funciones a los botones
        self.AcceptBtn.bind(on_release=self.accept)
        self.CancelBtn.bind(on_release=self.Cancel)
        self.clearnBtn.bind(on_release=self.clear_canvas)
        self.ButtonRED.bind(on_release=self.changeColorRED)
        self.ButtonBLUE.bind(on_release=self.changeColorBLUE)
        self.ButtonGREEN.bind(on_release=self.changeColorGREEN)
        self.ButtonUNDO.bind(on_release=self.painter.undo_draw)
        self.ButtonIncreaseSize.bind(on_release=self.increaseSize)
        self.ButtonDecreaseSize.bind(on_release=self.decreaseSize)

        #Añadir los botones y el paintWindow a la ventana principal
        self.rootWindow.add_widget(self.PaintWindow)
        self.rootWindow.add_widget(self.AcceptBtn)
        self.rootWindow.add_widget(self.CancelBtn)
        self.rootWindow.add_widget(self.clearnBtn)
        self.rootWindow.add_widget(self.ButtonRED)
        self.rootWindow.add_widget(self.ButtonBLUE)
        self.rootWindow.add_widget(self.ButtonGREEN)
        self.rootWindow.add_widget(self.ButtonUNDO)
        self.rootWindow.add_widget(self.ButtonIncreaseSize)
        self.rootWindow.add_widget(self.ButtonDecreaseSize)
        self.rootWindow.add_widget(self.Size)

        #Llama a la funcióon update
        kivy.clock.Clock.schedule_interval(self.update, 1)
        #Devuelve el rootWindow que sera la ventana que se abrira al iniciar el editor
        return self.rootWindow

    def on_stop(self):
        #Reinicia los elementos de la ventana y la cierra
        self.PaintWindow.clear_widgets()
        Window.close()
        reset()

    def update(self, *args):
        #En cada update actualiza el número del canvas que se refiere al tamaño de la brocha
        self.Size.text = str(size)

    def accept(self, obj):
        #Acciones que realizará cuando se de al boton de aceptar
        global objects
        global onEditMode
        global canvas
        global img
        global container
        global tkimg
        global size
        onEditMode=True
        #Vuelve las variables a su vaalor original
        objects=[]
        size = 1
        global color
        color = RED
        #Guarda un png del PaintWindow que tiene la imagen de fondo y lo que hemos coloreado
        self.PaintWindow.export_as_image().save("Edit.png")
        #carga la imagen nuevamente y sustituye la antigua
        img=Image.open("Edit.png")
        #Muestra La imagen en el panel principal de la interfaz
        canvas_x_padding = 100
        canvas_y_padding = 100
        kk = img
        if img.width > (windWidth - canvas_x_padding):
            new_width = windWidth - canvas_x_padding
            kk = img.resize((new_width, int(img.height * new_width / img.width)))
            #Ajusta el alto a la celda del panel principal

        elif img.height > (windHeight - canvas_y_padding):
            new_height = windHeight - canvas_y_padding
            kk = kk.resize((int(kk.width * new_height / kk.height), new_height))
            #Ajusta el ancho a la celda del panel principal

        #Convierte la imagen a la libreria Tkinter y lo cambia en la interfaz
        tkimg = ImageTk.PhotoImage(kk)
        canvas.config(width=img.width, height=img.height)
        canvas.itemconfig(container, image=tkimg)

        Window.close()

    def Cancel(self, obj):
        global objects
        objects = []
        #Devuelve la brocha a su tamaño y color original y cierra la ventana
        global size
        size = 1
        global color
        color = RED
        Window.close()

    def clear_canvas(self, obj):
        global objects
        objects = []
        self.painter.canvas.clear()

    def changeColorRED(self, obj):
        #Cambia a el color a rojo y deja presionado el botón
        global color
        self.ButtonRED.state = "down"
        self.ButtonBLUE.state = "normal"
        self.ButtonGREEN.state = "normal"
        color = RED

    def changeColorGREEN(self, obj):
        #Cambia a el color a verde y deja presionado el botón
        global color
        self.ButtonRED.state = "normal"
        self.ButtonBLUE.state = "normal"
        self.ButtonGREEN.state = "down"
        color = GREEN

    def changeColorBLUE(self, obj):
        #Cambia a el color a azul y deja presionado el botón
        self.ButtonRED.state = "normal"
        self.ButtonBLUE.state = "down"
        self.ButtonGREEN.state = "normal"
        global color
        color = BLUE

    def increaseSize(self, obj):
        #Aumenta el tamaño de la brocha
        global size
        if size < 25:
            size = size + 1

    def decreaseSize(self, obj):
        #Decrementa el tamaño de la brocha
        global size
        if size > 1:
            size = size - 1


def preProccessImage():
    global file
    # Si el archivo está cargado llama a processImagen
    if file:
        proccessImage()


def proccessImage():
    tones = 3
    global file

    # Carga la imagen a procesar

    imgOr = cv.imread(file,  cv.IMREAD_GRAYSCALE)
    imgselec= cv.imread('Edit.png')

    def selecTones():
        atones = []
        mask = imgselec[:, :, 0] == 255
        selecR = imgOr[mask]
        atones.append(set(selecR))
        print("R min: ", np.min(selecR), " | max: ", np.max(selecR))
        mask = imgselec[:, :, 1] == 255
        selecG = imgOr[mask]
        selecG = list(set(selecG) - set(selecR))
        atones.append(set(selecG))
        print("G min: ", np.min(selecG), " | max: ", np.max(selecG))
        mask = imgselec[:, :, 2] == 255
        selecB = imgOr[mask]
        selecB = list(set(selecB) - set(selecG))
        atones.append(set(selecB))
        print("B min: ", np.min(selecB), " | max: ", np.max(selecB))
        return atones
        #

    # Solo se pueden poner 3 colores (3 tonos). Para un numero de tonos modificable hay que meter el trozo de codigo respectivo
    # a un color dentro de un bucle y pasarle como parametro la lista de colores usada (de oscuro a claro)
    def selecTonesMulti():
        atones = []

        # blue
        mask = imgselec == (0, 0, 255)
        mask = np.logical_and(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        selecB = list(imgOr[mask])
        atones.append(set(selecB))
        minB = np.min(selecB) if len(selecB) > 0 else 0
        maxB = np.max(selecB) if len(selecB) > 0 else 0
        print("B min: ", minB, " | max: ", maxB)

        # green
        mask = imgselec == (0, 255, 0)
        mask = np.logical_and(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        selecG = imgOr[mask]
        selecG = selecG[selecG > maxB]
        atones.append(set(selecG))
        minG = np.min(selecG) if len(selecG) > 0 else 0
        maxG = np.max(selecG) if len(selecG) > 0 else 0
        print("B min: ", minG, " | max: ", maxG)

        # red
        mask = imgselec == (255, 0, 0)
        mask = np.logical_and(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        selecR = imgOr[mask]
        selecR = selecR[selecR > maxG]
        atones.append(set(selecR))
        minR = np.min(selecR) if len(selecR) > 0 else 0
        maxR = np.max(selecR) if len(selecR) > 0 else 0
        print("B min: ", minR, " | max: ", maxR)

        return atones
        #

    # Busca los marcadores dado el histograma de la imagen, el numero de tonos diferentes que puede
    # tener la imagen aparte del fondo y el umbral (threshold) a partir del cual empieza el foreground
    def findSeeds(hist, numTones=3):
        # Encuentra los maximos locales que existen en el histograma para detectar que tonos son los diferenciadores
        locmax = argrelextrema(hist, np.greater)[0]
        locmax
        marks = []
        print(locmax)

        # Segun el parametro /tones/, de todos los maximos locales se escogen unos cuantos los mas separados
        # entre ellos que se pueda
        for i in range(numTones):
            if i == 0:
                prev_index = 0
            else:
                prev_index = int(locmax.size * i / numTones)

            if i == numTones - 1:
                index = int(locmax.size * (i + 1) / numTones)
                part = locmax[prev_index:]
            else:
                index = int(locmax.size * (i + 1) / numTones)
                part = locmax[prev_index:index]

            # Parte el histograma del foreground en tantas partes como tonos haya y escoge el que le toca segun el bucle for
            histpart = np.transpose(np.array([hist[i] for i in part]))
            # Encuentra el maximo dentro de ese trozo del histrograma
            index_max_hist_part = int(np.where(histpart == np.max(histpart))[1][0])
            # Determina que tono corresponde al maximo selecionado
            maxlocal = locmax[int(index_max_hist_part + i * locmax.size / numTones)]
            # Lo agrega a las semillas
            marks.append(maxlocal)
        return marks

    # Ejecuta el random walker dados los datos de entrada (imagen B/N) y los marcadores.
    # Luego los muestra por pantalla mediante un grafico.
    def plot_random_walker(data, markers):
        # Ejecuta el random walker
        labels = random_walker(data, markers, beta=10, mode='bf')

        # Mostrar resultados en graficos
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                            sharex=True, sharey=True)
        ax1.imshow(data, cmap='gray')
        ax1.axis('off')
        ax1.set_title('Data')
        ax2.imshow(markers, cmap='magma')
        ax2.axis('off')
        title = 'Markers, tones=' + str(tones)
        ax2.set_title(title)
        ax3.imshow(labels, cmap='gray')
        ax3.axis('off')
        ax3.set_title('Segmentation')

        fig.tight_layout()
        plt.show()

    def triparticion_manual(markers):
        # Marcar background 1 (el threshold este se escogeria mediante muestras)
        mask1 = 10 < imgOr
        mask2 = imgOr < 30
        mask = np.logical_and(mask1, mask2)  # Escoge los pixeles entre los dos humbrales
        markers[mask] = 1
        # Marcar background 2
        mask1 = 100 < imgOr
        mask2 = imgOr < 120
        mask = np.logical_and(mask1, mask2)  # Escoge los pixeles entre los dos humbrales
        markers[mask] = 2
        # Marcar foreground mediante threshold
        mask1 = 200 < imgOr
        mask2 = imgOr < 210
        mask = np.logical_and(mask1, mask2)  # Escoge los pixeles entre los dos humbrales
        markers[mask] = 3

    def ProccessImage():

        markers = np.zeros(imgOr.shape, dtype=float)

        hist = cv.calcHist([imgOr], [0], None, [256], [0, 256])

        # marks = findSeeds(hist, numTones=tones)
        # print(marks)
        atones = selecTonesMulti()
        print(atones)
        # Recorrer cada uno de los grupos de tonos
        for i in range(len(atones)):
            # Recorrer cada tono marcando la semilla
            for tone in atones[i]:
                markers[imgOr == tone] = i + 1

        print(markers)

        # Marcar foreground mediante semillas. Marca cada pixel que coincida en color con el marcador como semilla.

        # Muestra por pantalla los resultados
        print("mostrando...")
        plot_random_walker(imgOr, markers.astype('uint'))
        #

    if cv.haveImageReader("Edit.png") & cv.haveImageReader(file):
        ProccessImage()
    # selecTonesMulti()
    # Threshold basico para comparar resultados de binarizacion
    # _, thres = cv.threshold(data, 120, 255, cv.THRESH_BINARY)
    # cv.imshow("image threshold", thres)
    # cv.waitKey(0)


def main():
    root.title('Random Walker Algorithm')
    root.geometry(f'{windWidth}x{windHeight}')
    root.resizable(False, False) #Evita que el usuario pueda cambiar la ventana de tamaño

    buttonFrame = tk.Frame(root)  # Agregar frame
    # Agregar canvas

    # Agregar botones
    btOpen = tk.Button(buttonFrame, text='Open', width=20,
                       command=lambda: open_file())  # Boton de abrir imagen
    btEdit = tk.Button(buttonFrame, text='Edit', width=20, command=lambda: start_edit())  # Boton de Editar imagen
    btProcess = tk.Button(buttonFrame, text='Process', width=20,
                          command=lambda: preProccessImage())  # Boton de procesar imagen

    buttonFrame.pack()
    btOpen.grid(
        row=0,
        column=0,
        padx=10,
        pady=10
    )
    btEdit.grid(
        row=0,
        column=1,
        padx=10,
        pady=10
    )
    btProcess.grid(
        row=0,
        column=2,
        padx=10,
        pady=10
    )
    canvas.pack()

    root.mainloop()


if __name__ == '__main__':
    root = tk.Tk()

    canvas = tk.Canvas(root, width=0, height=0, bg='black')
    container = canvas.create_image(0, 0, anchor='nw')
    main()
