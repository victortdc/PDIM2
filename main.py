from funcoes import *

if __name__ == '__main__':
    imagem_original = cv2.imread("imagens/cerebro.png", 0)  # abre a imagem
    ruido = salt_and_paper(imagem_original, 5000)  # faz o ruido
    imagem_mediana = filtroMediana(ruido)  # aplica o filtro da mediana
    imagem_highboost = filtroHighBoost(imagem_original, ruido, 1.1)  # aplica o filtro do high boost
    higboost_equalizado = histeq(imagem_highboost)
    # configuração do histograma
    cols = ['ImgOriginal', 'ImgNoise', 'ImgMediana', 'ImgHighBoost', 'HighBoostEqualizado']
    imgs = [imagem_original, ruido, imagem_mediana, imagem_highboost, higboost_equalizado]
    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0, i].imshow(imgs[i], cmap='gray')
        axs[1, i].hist(imgs[i].ravel(), bins=25, range=[0, 256])
        axs[0, i].set_title(cols[i])

    # print dos PSNR da imagem
    print("PSNR ruidosa:" + str(calculaPSNR(imagem_original, ruido)))
    print("PSNR mediana:" + str(calculaPSNR(imagem_original, imagem_mediana)))
    print("PSNR highboost:" + str(calculaPSNR(imagem_original, imagem_highboost)))
    print("PSNR highboost Equalizado:" + str(calculaPSNR(imagem_original, higboost_equalizado)))

    plt.show()
