import numpy as np
import sys
ENTRADA = np.array(
    [
        [1., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 1.],
    ]
)

Kernel1 = np.array([[0, -1, 0], [-1, 1, -1], [0, -1, 0]])

Kernel2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# Funcion que aplica el kernel a la imagen
def aplicar_kernel(imagen, kernel, padding=1, stride=1):
    # Calculo el tamaño de la imagen
    alto = imagen.shape[0]
    ancho = imagen.shape[1]

    # Calculo el tamaño de la salida
    alto_salida = int((alto - kernel.shape[0] + 2 * padding) / stride + 1)
    ancho_salida = int((ancho - kernel.shape[1] + 2 * padding) / stride + 1)

    # expando la imagen con padding
    imagen_expandida = np.zeros((alto + 2 * padding, ancho + 2 * padding))
    imagen_expandida[padding : alto + padding, padding : ancho + padding] = imagen

    # Creo la matriz de salida
    salida = np.zeros((alto_salida, ancho_salida))

    # Aplico el kernel
    for i in range(alto_salida):
        for j in range(ancho_salida):
            # Calculo el valor de la salida
            salida[i, j] = np.sum(
                imagen_expandida[i * stride : i * stride + kernel.shape[0], j * stride : j * stride + kernel.shape[1]]
                * kernel
            )

    return salida

if (len(sys.argv)<2):
    print("Ingresar stride")
    exit(1)

stride = int(sys.argv[1])

print("Imagen original")
print(ENTRADA)
print("Imagen con el kernel 1")
print(aplicar_kernel(ENTRADA, Kernel1,stride=stride))
print("Imagen con el kernel 2")
print(aplicar_kernel(ENTRADA, Kernel2,stride=stride))
