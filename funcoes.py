import numpy as np
import cv2
import matplotlib.pyplot as plt


def negativo(imagem):
    return 1 - imagem


def calculaMSE(imagem_original, imagem_ruidosa):
    soma_dif = 0
    for i in range(imagem_original.shape[0]):
        for j in range(imagem_ruidosa.shape[1]):
            soma_dif += float((float(imagem_original[i, j]) - float(imagem_ruidosa[i, j])) ** 2)

    return soma_dif / (imagem_original.shape[0] * imagem_original.shape[1])


def calculaPSNR(imagem_original, imagem_ruidosa):
    return np.max(np.log10(((np.max(imagem_original) ** 2) / calculaMSE(imagem_original, imagem_ruidosa)))) * 10


def salt_and_paper(imagem, intensidade_min=6000, intensidade=10000):
    imagem_ruidosa = imagem.copy()  # copia da imagem original

    # dimensões da imagem
    row, col = imagem_ruidosa.shape

    # espalha pixels brancos
    numero_de_pixels = np.random.randint(intensidade_min, intensidade)
    for i in range(numero_de_pixels):
        x_coord = np.random.randint(0, row)
        y_coord = np.random.randint(0, col)
        imagem_ruidosa[x_coord][y_coord] = 255

    # espalha pixels pretos
    numero_de_pixels = np.random.randint(intensidade_min, intensidade)
    for i in range(numero_de_pixels):
        x_coord = np.random.randint(0, row)
        y_coord = np.random.randint(0, col)
        imagem_ruidosa[x_coord][y_coord] = 0

    return imagem_ruidosa


def filtroPassaAltas(imagem_ruidosa, intensidade):
    mask_size = imagem_ruidosa.shape[0] * imagem_ruidosa.shape[1]
    w = (mask_size * intensidade) - 1
    masc = np.array([[-1, -1, -1],
                     [-1, w, -1],
                     [-1, -1, -1]], dtype="float")  # mascara passa-alta
    result = imagem_ruidosa.copy()
    for i in range(1, result.shape[0] - 2) :
        for j in range(1, result.shape[1] - 2) :
            result[i,j] = (
                result[i - 1, j - 1] * masc[0, 0]+
                result[i - 1, j] * masc[0, 1] +
                result[i - 1, j + 1]* masc[0, 2] +
                result[i, j - 1] * masc[1, 0] +
                result[i, j] * masc[1, 1] +
                result[i, j + 1] * masc[1, 2] +
                result[i + 1, j - 1] * masc[2, 0] +
                result[i + 1, j] * masc[2, 1] +
                result[i + 1, j + 1] * masc[2, 2]
            ) / mask_size
    return result


def filtroHighBoost(imagem, imagem_ruidosa, intensidade):
    passaAltas = filtroPassaAltas(imagem_ruidosa, intensidade)
    #return (intensidade - 1) * imagem.copy() + passaAltas
    return ((1 + intensidade) * imagem) - (intensidade * imagem_ruidosa)


def filtroMediana(image):
    result = image.copy()
    for i in range(1, result.shape[0] - 2):
        for j in range(1, result.shape[1] - 2):
            result[i, j] = np.sort(
                [result[i + 1, j],
                 result[i, j],
                 result[i - 1, j],
                 result[i + 1, j - 1],
                 result[i, j - 1],
                 result[i - 1, j - 1],
                 result[i + 1, j + 1],
                 result[i, j + 1],
                 result[i - 1, j + 1]]
            )[4]
    return result
