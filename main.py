from funcoes import *

if __name__ == '__main__':
    imagem_original = cv2.imread("imagens/cerebro.png", 0)  # abre a imagem

    ruido = salt_and_paper(imagem_original)

    # configuração do histograma
    cols = ['ImgOriginal', 'ImgNoise', 'ImgMediana']
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(imagem_original, cmap='gray')
    axs[1, 0].hist(imagem_original.ravel(), bins=25, range=[0, 256])
    ruido = salt_and_paper(imagem_original, 5000)  # faz o ruido
    axs[0, 1].imshow(ruido, cmap='gray')
    axs[1, 1].hist(ruido.ravel(), bins=25, range=[0, 256])

    imagem_mediana = filtroMediana(ruido)  # aplica o filtro da mediana
    imagem_highboost = filtroHighBoost(imagem_original, 1)  # aplica o filtro do high boost

    cv2.imshow("1", imagem_highboost)
    cv2.waitKey(0)

    # configuração do histograma
    axs[0, 2].imshow(imagem_mediana, cmap='gray')
    axs[1, 2].hist(imagem_mediana.ravel(), bins=25, range=[0, 256])

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    # print dos PSNR da imagem
    print("PSNR ruidosa:" + str(calculaPSNR(imagem_original, ruido)))
    print("PSNR mediana:" + str(calculaPSNR(imagem_original, imagem_mediana)))
    print("PSNR highboost:" + str(calculaPSNR(imagem_original, imagem_highboost)))

    plt.show()
