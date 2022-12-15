import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize(img, nivel=255.):
    _min = np.amin(img)
    _max = np.amax(img)
    result = (img - _min) * nivel / (_max - _min)

    return np.uint8(result)


def speckle(img):
    saida = img.copy()
    mask = np.random.rand(saida.shape[0], saida.shape[1])

    return normalize(saida * mask, 255.)


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


def filtroMediana(image):
    result = image.copy()
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
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


def contornoThresh(img):
    # Converte para RGB
    image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Cria imagem binária de threshold
    # , binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARYINV)
    _, binario = cv2.threshold(gray,0,225,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Encontre os contornos da imagem com threshold
    contours, hierarchy = cv2.findContours(binario, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_contornos = str(len(contours))

    # Plotar contornos
    contorno = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return binario, contorno, num_contornos


def f(imagem):
    imagem_original = cv2.imread(f"imagens/{imagem}.png", 0)  # abre a imagem
    ruido = salt_and_paper(imagem_original, 5000)  # faz o ruido
    edges = cv2.Canny(imagem_original, 100, 200)
    imagem_mediana = filtroMediana(ruido)  # aplica o filtro da mediana
    binario, bordas, n_bordas = contornoThresh(imagem_original)

    # configuração do histograma
    cols = ['ImgOriginal', 'ImgNoise', 'ImgCanny']
    imgs = [imagem_original, ruido, edges]
    fig, axs = plt.subplots(2, 3)
    for i in range(3):
        axs[0, i].imshow(imgs[i], cmap='gray')
        axs[1, i].hist(imgs[i].ravel(), bins=25, range=[0, 256])
        axs[0, i].set_title(cols[i])

    # configuração do histograma
    cols = ['mediana', 'medianaCanny']
    imgs = [imagem_mediana, edges]
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        axs[0, i].imshow(imgs[i], cmap='gray')
        axs[1, i].hist(imgs[i].ravel(), bins=25, range=[0, 256])
        axs[0, i].set_title(cols[i])

    cols = ['binario', 'contornada']
    imgs = [binario, bordas]
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        axs[0, i].imshow(imgs[i])
        axs[1, i].hist(imgs[i].ravel(), bins=25, range=[0, 256])
        axs[0, i].set_title(cols[i])

    '''
    # print dos PSNR da imagem
    print(f"PSNR ruidosa: {str(calculaPSNR(imagem_original, ruido))}")
    print(f"PSNR mediana: {str(calculaPSNR(imagem_original, imagem_mediana))}")
    print(f"PSNR highboost: {str(calculaPSNR(imagem_original, imagem_highboost))}\n\n")
    '''
    plt.show()
