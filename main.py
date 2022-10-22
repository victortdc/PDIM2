import cv2
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


def salt_and_paper(imagem, intensidade_min=6000, intensidade=10000):
    imagem_ruidosa = imagem.copy() #copia da imagem original

    # dimensões da imagem
    col, row = imagem_ruidosa.shape

    #espalha pixels brancos
    numero_de_pixels = np.random.randint(intensidade_min, intensidade)
    for i in range(numero_de_pixels):
        y_coord = np.random.randint(0, row)
        x_coord = np.random.randint(0, col)
        imagem_ruidosa[y_coord][x_coord] = 255

    #espalha pixels pretos
    numero_de_pixels = np.random.randint(intensidade_min, intensidade)
    for i in range(numero_de_pixels):
        y_coord = np.random.randint(0, row)
        x_coord = np.random.randint(0, col)
        imagem_ruidosa[y_coord][x_coord] = 0

    return normaliza(imagem_ruidosa)

def filtroMediana(img) :
    aux = np.pad(img.copy(), pad_width=1, mode='constant', constant_values=0)
    result = img.copy()
    for i in range(1, aux.shape[0] - 2):
        for j in range(1, aux.shape[1] - 2):
            result[i,j] = np.sort([aux[i+1,j], aux[i,j], aux[i-1,j], aux[i+1,j-1], aux[i,j-1], aux[i-1,j-1],  aux[i+1,j+1], aux[i,j+1], aux[i-1,j+1]])[4]
    
    return result


if __name__ == '__main__':
    img = cv2.imread("imagens/BrainOriginal.png", 0)  # abre a imagem
    cols = ['ImgOriginal', 'ImgNoise', 'ImgMediana']
    fig, axs = plt.subplots(2, 3)
    axs[0,0].imshow(img, cmap='gray')
    axs[1,0].hist(img.ravel(), bins=25, range=[0, 256])

    noise = salt_and_paper(img, 5000)  # faz o ruido
    axs[0,1].imshow(noise, cmap='gray')
    axs[1,1].hist(noise.ravel(), bins=25, range=[0, 256])

    img_median = filtroMediana(noise) #aplica o filtro
    axs[0,2].imshow(img_median, cmap='gray')
    axs[1,2].hist(img_median.ravel(), bins=25, range=[0, 256])

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    
    img_median_merda = cv2.medianBlur(noise, 3)  # aplica o filtro

    cv2.imshow('manual', img_median)
    cv2.imshow('pronta', img_median_merda)

    print("PSNR ruidosa:" + str(calculaPSNR(img,noise)))
    print("PSNR mediana(manual):" + str(calculaPSNR(img,img_median)))
    print("PSNR mediana(pronta):" + str(calculaPSNR(img,img_median_merda)))
    
    plt.show()

