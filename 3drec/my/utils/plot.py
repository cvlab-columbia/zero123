import numpy as np
import matplotlib.pyplot as plt


def mpl_fig_to_buffer(fig):
    fig.canvas.draw()
    plot = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return plot
