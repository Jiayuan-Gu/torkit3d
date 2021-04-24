import matplotlib.pyplot as plt


def show_image(image, show_axis=False):
    plt.cla()
    plt.axis('on' if show_axis else 'off')
    if image.ndim == 2:
        plt.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    else:
        plt.imshow(image)
    plt.show()
