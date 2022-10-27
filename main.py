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


    def imhist(im):
        m, n = im.shape
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[im[i, j]] += 1
        return np.array(h) / (m * n)


    def cumsum(h):
        return [sum(h[:i + 1]) for i in range(len(h))]


    def histeq(im):
        # calculate Histogram
        h = imhist(im)
        cdf = np.array(cumsum(h))
        sk = np.uint8(255 * cdf)
        s1, s2 = im.shape
        Y = np.zeros_like(im)

        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[im[i, j]]
        H = imhist(Y)

        return Y, h, H, sk


    def histogramEq(test):

        img = imagem_highboost
        new_img, h, new_h, sk = histeq(img)

        fig = plt.figure()
        fig.add_subplot(221)
        plt.plot(h)
        plt.title('Original histogram')  # original histogram

        fig.add_subplot(222)
        plt.plot(new_h)
        plt.title('New histogram')  # hist of eqlauized image

        fig.add_subplot(223)
        plt.plot(sk)
        plt.title('Transfer function')  # transfer function

        cv2.imwrite('histogramEq.jpg', new_img)
        cv2.imshow

    histogramEq(imagem_highboost)
    file = open("histogramEq.jpg", "rb")
    image = file.read()
    new_image = cv2.imread("histogramEq.jpg")
    print("After: ", new_image.shape[0], " x ", new_image.shape[1], end="\r")
    # print dos PSNR da imagem
    print("PSNR ruidosa:" + str(calculaPSNR(imagem_original, ruido)))
    print("PSNR mediana:" + str(calculaPSNR(imagem_original, imagem_mediana)))
    print("PSNR highboost:" + str(calculaPSNR(imagem_original, imagem_highboost)))

    plt.show()
