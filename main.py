import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


def normaliza(imagem):
    _min = np.amin(imagem)
    _max = np.amax(imagem)
    result = (imagem - _min) * 255.0 / (_max - _min)

    return np.uint8(result)


def calculaMSE(imagem_original, imagem_ruidosa):
    soma_dif = 0
    for i in range(imagem_original.shape[0]):
        for j in range(imagem_ruidosa.shape[1]):
            soma_dif += float((float(imagem_original[i, j]) - float(imagem_ruidosa[i, j])) ** 2)

    return soma_dif / (imagem_original.shape[0] * imagem_original.shape[1])


def calculaPSNR(imagem_original, imagem_ruidosa):
    return np.max(np.log10(((np.max(imagem_original) ** 2) / calculaMSE(imagem_original, imagem_ruidosa)))) * 10


def salt_and_paper(imagem, intensidade=1000):
    imagem_ruidosa = imagem.copy()  # copia da imagem original

    # dimensões da imagem
    col, row = imagem_ruidosa.shape

    # espalha pixels brancos
    numero_de_pixels = random.randint(0, intensidade)
    for i in range(numero_de_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)

        imagem_ruidosa[y_coord][x_coord] = 255

    # espalha pixels pretos
    numero_de_pixels = random.randint(0, intensidade)
    for i in range(numero_de_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)

        imagem_ruidosa[y_coord][x_coord] = 0

    return normaliza(imagem_ruidosa)


if __name__ == '__main__':
    img = cv2.imread("imagens/BrainOriginal.png", 0)  # abre a imagem
    plt.hist(img.ravel(), bins=25, range=[0, 256], label="original", alpha=.8)
    plt.title('Histogram for gray scale image')
    plt.legend()
    plt.show()
    cv2.imshow('ImgOriginal', img)

    noise = salt_and_paper(img, 5000)  # faz o ruido
    plt.hist(noise.ravel(), bins=25, range=[0, 256], label="SaltyPaper", alpha=.8)
    plt.title('Histogram for gray scale image')
    plt.legend()
    plt.show()
    cv2.imshow('ImgNoise', noise)

    img_median = cv2.medianBlur(noise, 5)  # aplica o filtro
    plt.hist(img_median.ravel(), bins=25, range=[0, 256], label="original", alpha=.8)
    plt.title('Histogram for gray scale image')
    plt.legend()
    plt.show()
    cv2.imshow('ImgMediana', img_median)
    cv2.waitKey(0)
