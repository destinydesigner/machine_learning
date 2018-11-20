import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')


def plot_3d_points(l):
    """
    https://matplotlib.org/mpl_toolkits/mplot3d/index.html

    display draggable image:
        1. please disable *Settings | Tools | Python Scientific | Show plots in toolwindow*.
        2. https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

    :param l:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*zip(*l), c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == '__main__':
    from random import random
    plot_3d_points(zip(
        [random() for _ in range(10)],
        [random() for _ in range(10)],
        [random() for _ in range(10)],
    ))
