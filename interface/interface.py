import tkinter
from random import randint
from PIL import Image, ImageTk
import os
import model.model as mm


class App:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title("Railway station")

        # создаем рабочую область
        self.frame = tkinter.Frame(self.root)
        self.frame.grid()

        # вставляем кнопку
        self.but = tkinter.Button(self.frame,
                                  text="Get graph image",
                                  command=self.button_click,
                                  activeforeground="#555555").grid(row=0, column=0)

        # Добавим изображение
        self.canvas = tkinter.Canvas(self.root, height=600, width=700)
        self.canvas.grid(row=1, column=0)
        self.root.mainloop()

    @staticmethod
    def get_graph_image():
        # Function that draws graph image
        graph = mm.Graph(4, edges=[[0, 1], [2, 3], [2, 1], [0, 3], [0, 2]],
                         cities=["Moscow", "New York", "Tokyo", "Rome"], directed=True)
        graph.set_distances_between_cities([100, 2000, 3, 5, 1])
        graph.draw_graph()

    def button_click(self):
        self.get_graph_image()
        self.image = Image.open("image.png")
        self.photo = ImageTk.PhotoImage(self.image)
        self.c_image = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.grid(row=1, column=0)

app = App()