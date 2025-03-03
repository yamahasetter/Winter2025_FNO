"""
    This is to make an animation
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.colors import SymLogNorm
import numpy as np
import os


def make_movie(Z, num_time_pts, dt, title = 'movie', colors = 'RdBu_r', dir=os.getcwd()):
    # This is the colormap I'd like to use.
    # cm = plt.cm.get_cmap('YlGnBu')
    binary_cmap = ListedColormap(['red', 'blue'])
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1, clip=False)

    # make figure
    fig, ax = plt.subplots()
    # cbar = plt.colorbar(ax.contourf(Z[0], cmap="RdYlBu"))

    def animate(n):
        print('running step ', n)
        ax.cla()

        plt.pcolormesh(Z[n], norm=SymLogNorm(linthresh=0.09, linscale=1, vmin=-1, vmax=1),cmap=colors)
        # ax.contourf(Z[n], cmap=colors)

        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        ax.set_aspect('equal')

        ax.set_title(r"$t=$" + str(round(dt*n,5)))

        return fig,


    # FuncAnimation(fig = fig, func = animate, frames = 10, interval = 1, repeat = False)

    anim = FuncAnimation(fig = fig, func = animate, frames = num_time_pts, interval = 1, repeat = False)
    anim.save(dir + '/' + title+'.mp4',fps=30)