from lib2to3.pgen2.token import STAR
from tkinter import *
from tkinter import ttk
from tkinter import font
import numpy as np
import math
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from idlelib.tooltip import Hovertip
# Importo Middeleware para tratar las salidas de la gui
import middleware_gui as mw
# importo funciones de activacion
import funciones_act as fa
# Funcion para cargar el archivo
from dataset_gen import distortion_pattern
# Componentes de la red neuronal
from neural_net_components import create_neural_net
# Funcion para entrenar la red y predecir
from neural_net_functions import predict, training


# ------------------------------------------------------------
ventana = Tk()  # Ventana principal
ventana.title("MLP")  # titulo de la ventana
ventana.configure(background='gray')  # color de fondo

#-------------------    FUENTES    -------------------#
fontTitle = font.Font(family="Arial", size=14, weight="bold", slant="italic")
fontTitle2 = font.Font(family="Arial", size=12, weight="bold", slant="italic")
fontButton = font.Font(family="Arial", size=10, weight="bold", slant="italic")
fontLabel = font.Font(family="Arial", size=10, weight="bold", slant="italic")
fontBox = font.Font(family="Arial", size=8, weight="bold", slant="italic")
fontResult = font.Font(family="Arial", size=6)

#-------------------    PACKAGE    -------------------#
package = Frame(ventana, width=600, height=600, bg="gray", bd=10,
                relief="ridge")  # relief es el marco del package
# stick alinia el contenido de la celda - nsew = north south east west
package.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

package_options = Frame(package, width=600, height=600, bg="gainsboro")
package_options.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

package_options1 = Frame(package, width=600, height=600, bg="gainsboro")
package_options1.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

package_matriz = Frame(package, width=600, height=600, bg="gray")
package_matriz.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

#--------------------- CANVAS ---------------------#
#---------------   VALORES INICIALES --------------#
# Tamaño de la matriz
w = 10
h = 10
input_size = w * h
states = np.zeros((w, h))
rect_size = 50  # Tamaño de los botones/cuadros de la matriz

# Se inicia en blanco una matriz de tamaño W x H en forma de arreglo, mas adelante se guarda la letra dibujada
array_resultado = np.zeros((w, h))

# Al hacer click en una cuadricula pinta o despinta dependiendo su estado anterior


def mouseClick(event):
    x = math.floor(event.x / rect_size)
    y = math.floor(event.y / rect_size)
    if x < w and y < h:
        # Cambia entre 0 y 1 dependiendo si esta marcando o no
        states[x, y] = 0 if states[x, y] > 0 else 1
    print_grid()


# Crea el Canvas dentro de package_matriz
canvas = Canvas(package_matriz, width=rect_size*w,
                height=rect_size*h, cursor="spraycan")
canvas.bind("<Button-1>", mouseClick)
canvas.pack(fill=X, pady=2)

# Dibuja los cuadrados de la matriz


def print_grid():
    for i in range(w):
        for j in range(h):
            color = 'green' if states[i, j] > 0 else 'white'
            canvas.create_rectangle(i * rect_size, j * rect_size, (i + 1) * rect_size,
                                    (j + 1) * rect_size, outline="black", fill=color)


print_grid()

# -------------------    Pinto el patrón distorcionado    -------------------#


def print_grid_dist():
    global patron_distorsionado
    distorcion = float(box8.get())
    patron_original = np.array(mw.getPatron(box7.get()))
    patron_distorsionado = np.array(
        distortion_pattern(patron_original, distorcion))  # Aca ver despues el tipo de datos
    for i in range(w):
        for j in range(h):
            states[j, i] = patron_distorsionado[i][j]
    print_grid()

# -------------------    Limpio states y resultados  -------------------#


def clear_callback():
    np.ndarray.fill(states, 0)
    np.ndarray.fill(array_resultado, 0)
    print_grid()


#--------------------- MENU ---------------------#
package_lore = Frame(package_options, bg="gainsboro")
package_lore.grid(row=0, padx=10, pady=5)
etiquetaTitulo = Label(package_lore, text="Entrenamiento",
                       bg="gainsboro", fg="black", font=fontTitle)
etiquetaTitulo.grid(row=0, column=0, padx=10, pady=2)

etiquetaDescripcion = Label(package_lore, text="Seleccione los parámetros para entrenar la red",
                            bg="gainsboro", fg="black", font=fontLabel)
etiquetaDescripcion.grid(row=1, column=0, padx=10, pady=2)


package_lore = Frame(package_options1, bg="gainsboro")
package_lore.grid(row=0, padx=10, pady=5)
etiquetaTitulo = Label(package_lore, text="Inferencia",
                       bg="gainsboro", fg="black", font=fontTitle)
etiquetaTitulo.grid(row=0, column=0, padx=10, pady=2)


#-------------------    OPCIONES    -------------------#
# Etiqueta 1
package_e1 = Frame(package_options, bg="gainsboro")
package_e1.grid(row=1, padx=10, pady=5)
etiqueta1 = Label(package_e1, text="Seleccione el Dataset",
                  bg="gainsboro", font=fontLabel)
etiqueta1.grid()
box1 = ttk.Combobox(package_e1,
                    state="readonly",  # Desactiva la caja de texto, solo deja activada la lista
                    values=["100_10validacion", "100_20validacion", "100_30validacion",
                            "500_10validacion", "500_20validacion", "500_30validacion",
                            "1000_10validacion", "1000_20validacion", "1000_30validacion"],
                    font=fontBox, cursor="hand2"
                    )
box1.grid()
box1.current(0)  # Selecciona por defecto el primer valor de la lista
# Etiqueta 2
package_e2 = Frame(package_options, bg="gainsboro")
package_e2.grid(row=2, padx=10, pady=5)
etiqueta2 = Label(package_e2, text="Cantidad de capas ocultas",
                  bg="gainsboro", font=fontLabel)
etiqueta2.grid()
box2 = ttk.Combobox(package_e2,
                    state="readonly",
                    values=["1", "2"],
                    font=fontBox, cursor="hand2"
                    )
box2.grid()
box2.current(0)
# Etiqueta 3
package_e3 = Frame(package_options, bg="gainsboro")
package_e3.grid(row=3, padx=10, pady=5)
etiqueta3 = Label(package_e3, text="Cantidad de Neuronas por capa",
                  bg="gainsboro", font=fontLabel)
etiqueta3.grid()
box3 = ttk.Combobox(package_e3,
                    state="readonly",
                    values=["5", "10"],
                    font=fontBox, cursor="hand2"
                    )
box3.grid()
box3.current(0)
# Etiqueta 4
package_e4 = Frame(package_options, bg="gainsboro")
package_e4.grid(row=4, padx=10, pady=5)
etiqueta4 = Label(package_e4, text="Coeficiente de aprendizaje = 0.5",
                  bg="gainsboro", font=fontLabel)
etiqueta4.grid()
# Etiqueta 6
package_e6 = Frame(package_options, bg="gainsboro")
package_e6.grid(row=6, padx=10, pady=5)
etiqueta6 = Label(package_e6, text="Término momento",
                  bg="gainsboro", font=fontLabel)
etiqueta6.grid()
box6 = ttk.Combobox(package_e6,
                    state="readonly",
                    values=["0.5", "0.9"],
                    font=fontBox, cursor="hand2"
                    )
box6.grid()
box6.current(0)

# Espacio para botones de entrenamiento
package_e7 = Frame(package_options, bg="gainsboro")
package_e7.grid(row=12, padx=10, pady=5)
# Boton de predicción
package_e11 = Frame(package_options1, bg="gainsboro")
package_e11.grid(row=12, padx=10, pady=5)
# Espacio para boton de limpiar
package_e8 = Frame(package_options1, bg="gainsboro")
package_e8.grid(row=11, padx=10, pady=5)
# Espacio para boton de cargar modelo
package_e14 = Frame(package_options, bg="gainsboro")
package_e14.grid(row=16, padx=10, pady=8)

# Espacio para etiqueta de carga de patrón
package_e9 = Frame(package_options1, bg="gainsboro")
package_e9.grid(row=9, padx=10, pady=5)
etiqueta9 = Label(package_e9, text="Seleccione el patrón a cargar y su distorsión",
                  bg="gainsboro", font=fontLabel)
etiqueta9.grid()
# Espacio para caja de texto de carga de patrón y distorsión
package_e10 = Frame(package_options1, bg="gainsboro")
package_e10.grid(row=10, padx=10, pady=1)
box7 = ttk.Combobox(package_e10,
                    state="readonly",
                    values=["Patron B", "Patron D", "Patron F"],
                    font=fontBox, cursor="hand2",
                    width=10
                    )
box7.grid(row=0, column=0, padx=10)
Hovertip(box7, text="Seleccione un patrón", hover_delay="5")
box7.current(0)

box8 = ttk.Combobox(package_e10,
                    state="readonly",
                    values=["0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"],
                    font=fontBox, cursor="hand2",
                    width=10
                    )
box8.grid(row=0, column=1, padx=10)
Hovertip(box8, text="Seleccione una distorsión", hover_delay="5")
box8.current(0)

# Espacio para etiqueta de cargar modelo entrenado
package_e12 = Frame(package_options, bg="gainsboro")
package_e12.grid(row=13, padx=10, pady=5)
etiqueta12 = Label(package_e12, text="O bien, seleccionar un modelo ya entrenado",
                   bg="gainsboro", font=fontLabel)
etiqueta12.grid()
# Espacio para caja de carga de modelo entrenado
package_e13 = Frame(package_options, bg="gainsboro")
package_e13.grid(row=14, padx=10, pady=1)
box9 = ttk.Combobox(package_e13,
                    state="readonly",
                    values=["100_10val_1CapOc_5neu_mom05", "100_20val_2CapOc_10neu_mom09",
                            "500_20val_2CapOc_10neu_mom05", "500_10val_1CapOc_5neu_mom09", "1000_30val_2CapOc_10neu_mom09", "1000_10val_1CapOc_5neu_mom05"],
                    font=fontBox, cursor="hand2", width=35
                    )
box9.grid(row=0, column=0)
box9.current(0)

#-------------------    FUNCIONES PARA BOTONES    -------------------#


def mouseClickCargarModelo():
    global banderaModelo
    banderaModelo = True
    print(banderaModelo)
    global trained_neural_net
    global error_validacion_ent
    global error_validacion_val
    global newWindows
    global mse
    global accuracy
    global mse_test
    newWindows = Toplevel(ventana)
    newWindows.title("Resultados")
    newWindows.geometry("500x500")
    # Obtengo el modelo
    data_trainning = mw.getModelo(box9.get())
    # Red neuronal entrenada
    trained_neural_net = data_trainning[0]
    # Precisión de la red
    accuracy = data_trainning[1]
    # Error del set de validación
    mse = data_trainning[2]
    # Error del set de test
    mse_test = data_trainning[3]

    # Error de validacion de entrenamiento
    error_validacion_ent = data_trainning[4]
    # Error de validacion de validacion
    error_validacion_val = data_trainning[5]
    # Trato los datos de accuracy y mse para mostrarlos en la ventana
    res = mw.ordenarMSEyAccuracy(mse, accuracy)

    #-------------------    Titulo    -------------------#
    label = Label(newWindows, text="Resultados",
                  bg="gainsboro", fg="black", font=fontTitle)
    label.pack(padx=10)
    #-------------------    Scroll    -------------------#
    scroll = Scrollbar(newWindows)
    scroll.pack(side=RIGHT, fill=Y)
    #-------------------    Lista de resultados    -------------------#
    mylist = Listbox(newWindows, yscrollcommand=scroll.set,
                     width=100, height=15, font=fontBox, justify='left', borderwidth=0)
    mylist.configure(background="gainsboro")
    mylist.insert(STAR, "Épocas" + " " * 10 +
                  "MSE de Validación" + " " * 20 + "Precisión")
    mylist.insert(
        STAR, "--------------------------------------------------------------------------------------")
    # Por cada elemento en el array de resultados, lo agrego a la lista
    for line in res:
        mylist.insert(END, line)
    # Configuro el scroll
    scroll.config(command=mylist.yview, cursor="hand2")
    # Muestro la lista y el scroll
    mylist.pack(padx=10, pady=10)
    # Label para precision de test
    test = Label(
        newWindows, text=f'MSE del conjunto de test: {mse_test}', font=fontBox)
    test.pack(padx=10, pady=10)
    #-------------------    Boton de graficar    -------------------#
    btnPres = Button(newWindows, text="Gráfico Precisión", bg="gainsboro",
                     fg="black", command=plotAccuracy, font=fontButton, cursor="hand2")
    btnErr = Button(newWindows, text="Gráfico Error de Validación ", bg="gainsboro",
                    fg="black", command=plotValidation, font=fontButton, cursor="hand2")
    btnMse = Button(newWindows, text="Gráfico MSE", bg="gainsboro",
                    fg="black", command=plotMSE, font=fontButton, cursor="hand2")
    btnPres.place(x=80, y=460)
    btnMse.place(x=290, y=460)
    btnErr.place(x=140, y=400)
    Hovertip(btnPres, text="Crea el gráfico de precisión del modelo.",
             hover_delay="5")
    Hovertip(btnErr, text="Crea el gráfico de del error de validación del conjunto de entrenamiento y validación.",
             hover_delay="5")
    Hovertip(
        btnMse, text="Crea el gráfico del MSE del conjunto de validación.", hover_delay="5")


def mouseClickEntrenar():
    global banderaModelo
    global trained_neural_net
    global newWindows
    global mse
    global accuracy
    global mse_test
    global error_validacion_ent
    global error_validacion_val
    banderaModelo = False
    print(banderaModelo)
    newWindows = Toplevel(ventana)
    newWindows.title("Resultados")
    newWindows.geometry("500x500")

    # Array de opciones
    array_opciones = [str(box1.get()), str(box2.get()), str(
        box3.get()), str(box6.get())]
    # Trato los datos
    # cargo el dataset
    dataset = mw.getDataset(array_opciones)
    # separo el dataset
    input_X = np.array(dataset[0])
    input_Y = np.array(dataset[1])
    entrada_test = np.array(dataset[2])
    salida_test = np.array(dataset[3])
    entrada_validacion = np.array(dataset[4])
    salida_validacion = np.array(dataset[5])
    # defino una topologia para la red
    # 100 entradas, 5 o 10 neuronas en la capa oculta, 5 o 10 neuronas en la capa oculta, 3 salidas
    topologia = mw.crear_arquitectura(array_opciones)
    # tipo de datos de topologia
    red_neuronal = create_neural_net(topologia, fa, layers=array_opciones[1])
    # Obtengo el momento
    momentum = mw.getMomentum(array_opciones)
    trained_neural_net = None
    data_trainning = training(100, red_neuronal, input_X, input_Y,
                              entrada_validacion, salida_validacion, entrada_test, salida_test, 0.5, momentum, fa, 0.5, 100)
    # Red neuronal entrenada
    trained_neural_net = data_trainning[0]
    # Precisión de la red
    accuracy = data_trainning[1]
    # Error del set de validación
    mse = data_trainning[2]
    # Error del set de test
    mse_test = data_trainning[3]

    # Error de validacion de entrenamiento
    error_validacion_ent = data_trainning[4]
    # Error de validacion de validacion
    error_validacion_val = data_trainning[5]

    # Trato los datos de accuracy y mse para mostrarlos en la ventana
    res = mw.ordenarMSEyAccuracy(mse, accuracy)

    #-------------------    Titulo    -------------------#
    label = Label(newWindows, text="Resultados",
                  bg="gainsboro", fg="black", font=fontTitle)
    label.pack(padx=10)
    #-------------------    Scroll    -------------------#
    scroll = Scrollbar(newWindows)
    scroll.pack(side=RIGHT, fill=Y)
    #-------------------    Lista de resultados    -------------------#
    mylist = Listbox(newWindows, yscrollcommand=scroll.set,
                     width=100, height=15, font=fontBox, justify='left', borderwidth=0)
    mylist.configure(background="gainsboro")
    mylist.insert(STAR, "Épocas" + " " * 10 +
                  "MSE de Validación" + " " * 20 + "Precisión")
    mylist.insert(
        STAR, "--------------------------------------------------------------------------------------")
    # Por cada elemento en el array de resultados, lo agrego a la lista
    for line in res:
        mylist.insert(END, line)
    # Configuro el scroll
    scroll.config(command=mylist.yview, cursor="hand2")
    # Muestro la lista y el scroll
    mylist.pack(padx=10, pady=10)
    # Label para precision de test
    test = Label(
        newWindows, text=f'MSE del conjunto de test: {mse_test}', font=fontBox)
    test.pack(padx=10, pady=10)
    #-------------------    Boton de graficar    -------------------#
    btnPres = Button(newWindows, text="Gráfico Precisión", bg="gainsboro",
                     fg="black", command=plotAccuracy, font=fontButton, cursor="hand2")
    btnErr = Button(newWindows, text="Gráfico Error de Validación ", bg="gainsboro",
                    fg="black", command=plotValidation, font=fontButton, cursor="hand2")
    btnMse = Button(newWindows, text="Gráfico MSE", bg="gainsboro",
                    fg="black", command=plotMSE, font=fontButton, cursor="hand2")
    btnPres.place(x=80, y=460)
    btnMse.place(x=290, y=460)
    btnErr.place(x=140, y=400)
    Hovertip(btnPres, text="Crea el gráfico de precisión del modelo.",
             hover_delay="5")
    Hovertip(btnErr, text="Crea el gráfico de del error de validación del conjunto de entrenamiento y validación.",
             hover_delay="5")
    Hovertip(
        btnMse, text="Crea el gráfico del MSE del conjunto de validación.", hover_delay="5")


#-------------------    RESULTADO    -------------------#
def mouseClickResultado():
    # Array de opciones
    array_opciones = [str(box1.get()), str(box2.get()), str(
        box3.get()), str(box6.get())]
    # Array de resultados
    for i in range(w):
        for j in range(h):
            array_resultado[i][j] = int(states[(j), (i)])
    # Trato los datos que toma de la grilla de 10x10
    entrada = mw.tratarEntrada(array_resultado)

    # --------------   Ventana    --------------#
    newWindows = Toplevel(ventana)
    newWindows.title("Resultados")
    newWindows.geometry("500x300")
    # --------------   Titulo    --------------#
    label = Label(newWindows, text="Resultados",
                  bg="gainsboro", fg="black", font=fontTitle)
    label.pack(padx=10)

    # Hago la prediccion con los datos de la grilla
    prediccion = predict(trained_neural_net, entrada)
    if len(prediccion[0]) == 1:
        # Trato los datos de la prediccion
        resultado1 = mw.tratarSalida(prediccion[0])
        resultado2 = mw.tratarSalida(prediccion[2])
        resultado3 = mw.tratarSalida(prediccion[4])
        # --------------   Label Letra   --------------#

        resultados = Label(
            newWindows, text=f'{resultado1}', bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=150, y=100, anchor="center")

        resultados = Label(
            newWindows, text=f'{resultado2}', bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=150, y=150, anchor="center")

        resultados = Label(
            newWindows, text=f'{resultado3}', bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=150, y=200, anchor="center")

        # --------------   Label Probabilidad   --------------#
        resultados = Label(
            newWindows, text=f'--->   {prediccion[1]}', bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=290, y=100, anchor="center")

        resultados = Label(
            newWindows, text=f'--->   {prediccion[3]}', bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=290, y=150, anchor="center")

        resultados = Label(
            newWindows, text=f'--->   {prediccion[5]}', bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=290, y=200, anchor="center")
    else:
        resultados = Label(newWindows, text=f'No se pudo reconocer la letra',
                           bg="gainsboro", fg="black", font=fontLabel)
        resultados.place(x=250, y=150, anchor="center")


# -------------- Funciones para graficar -------------- #
def plotMSE():
    error = []
    epocas = []
    for i in range(len(mse)):
        error.append(mse[i][1])
        epocas.append(mse[i][0])
    window = Toplevel(newWindows)
    window.title('Gráfico MSE de Validación')
    window.geometry("500x500")
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(epocas, error, color='green', marker='^', label='MSE')
    plot1.set_title("MSE del Conjunto de Validación", fontdict={
                    'fontsize': 14, 'fontweight': 'bold', 'color': 'tab:green'})
    plot1.set_xlabel("Épocas")
    plot1.set_ylabel("Error")
    plot1.legend(loc='upper right')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()
    canvas.get_tk_widget().pack()


def plotAccuracy():
    presicion = []
    epocas = []
    for i in range(len(accuracy)):
        presicion.append(accuracy[i][1])
        epocas.append(accuracy[i][0])
    window = Toplevel(newWindows)
    window.title('Gráfico Precisión')
    window.geometry("500x500")
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(epocas, presicion, color='purple', marker='^', label='PRES')
    plot1.set_title("Precisión del Modelo", fontdict={
                    'fontsize': 14, 'fontweight': 'bold', 'color': 'tab:purple'})
    plot1.set_xlabel("Épocas")
    plot1.set_ylabel("Precisión")
    plot1.legend(loc='upper right')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()
    canvas.get_tk_widget().pack()


def plotValidation():
    ent = []
    epocas_ent = []
    val = []
    epocas_val = []
    for i in range(len(error_validacion_ent)):
        ent.append(error_validacion_ent[i][1])
        epocas_ent.append(error_validacion_ent[i][0])
    for i in range(len(error_validacion_val)):
        val.append(error_validacion_val[i][1])
        epocas_val.append(error_validacion_val[i][0])
    window = Toplevel(newWindows)
    window.title('Gráfico Error de Validación')
    window.geometry("500x500")
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.plot(epocas_ent, ent, color='purple', marker='^', label='VAL_ENT')
    plot1.plot(epocas_val, val, color='green', marker='*', label='VAL_VAL')

    plot1.set_title("Error de Validación", fontdict={
                    'fontsize': 14, 'fontweight': 'bold', 'color': 'tab:blue'})
    plot1.set_xlabel("Épocas")
    plot1.set_ylabel("Error")
    plot1.legend(loc='upper right')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()
    canvas.get_tk_widget().pack()


# Etiqueta Entrenar
etiquetaEntrenar = Button(package_e7, text="Entrenar Modelo", bg="gainsboro",
                          fg="black", command=mouseClickEntrenar, font=fontButton, cursor="hand2")
etiquetaEntrenar.grid(row=0, column=0, padx=10)
Hovertip(etiquetaEntrenar,
         text="Entrena el modelo con los parametros seleccionados. Si elige otra arquitectura debe volver a entrenar.", hover_delay="5")
# Etiqueta Predecir
etiquetaPredecir = Button(package_e11, text="Predecir", bg="gainsboro",
                          fg="black", command=mouseClickResultado, font=fontButton, cursor="hand2")
etiquetaPredecir.grid(row=0, column=0, padx=10)
Hovertip(etiquetaPredecir,
         text="Predice el ejemplo cargado en la grilla.", hover_delay="5")
# Etiqueta limpiar grilla
clear = Button(package_e8, text="Limpiar Grilla", bg="gainsboro",
               fg="black", command=clear_callback, font=fontButton, cursor="hand2")
clear.grid(row=0, pady=10)
Hovertip(clear, text="Limpia la grilla devolviendola a su estado inicial.", hover_delay="5")
# Etiqueta cargar patrón
cargar = Button(package_e10, text="Cargar", bg="gainsboro",
                fg="black", command=print_grid_dist, font=fontButton, cursor="hand2")
cargar.grid(row=0, column=2, pady=5)
Hovertip(cargar, text="Carga el patrón de ejemplo configurado a la grilla. No olvide limpiar la grilla antes de cargar un patrón nuevo.", hover_delay="5")
# Etiqueta oara cargar el modelo
etiquetaCargar = Button(package_e14, text="Cargar Modelo", bg="gainsboro",
                        fg="black", command=mouseClickCargarModelo, font=fontButton, cursor="hand2")
etiquetaCargar.grid(row=0, column=0, padx=10)
Hovertip(etiquetaCargar,
         text="Carga el modelo seleccionado.", hover_delay="5")


ventana.mainloop()
