import matplotlib.pyplot as plt
import numpy

import analyze_util


class Visualizer(object):
    def __init__(self, field_size):
        self.field_size = field_size
        self.colormap = analyze_util.make2Dcolormap(size=self.field_size * 2)

    def save_png(self, f_base):
        self.fig.savefig(f_base + '.png')

    def save_eps(self, f_base):
        self.fig.savefig(f_base + '.eps')

    def save_pdf(self, f_base):
        self.fig.savefig(f_base + '.pdf', rasteriszed=True, dpi=300)

    def put_label(self, x_lab, y_lab, fontsize=32):
        plt.xlabel(x_lab, fontsize=fontsize)
        plt.ylabel(y_lab, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    def init_plot(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_rasterization_zorder(1)
        self.ax.tick_params(direction='in', length=10)

        plt.xticks(numpy.arange(-10, 10, 1))
        plt.yticks(numpy.arange(-10, 10, 1))

    def plot(
            self,
            hidden,
            position,
            plot_type,
            lim,
    ):

        ds = []
        for n in range(hidden.shape[0]):
            h = hidden[n, :]
            p = position[n, :]
            d = self.plot_hidden(h, p, plot_type=plot_type, lim=lim)
            ds.append(d)
        return ds

    def set_transparent(self):
        self.fig.patch.set_visible(False)
        self.ax.patch.set_visible(False)
        plt.axis('off')

    def close_plot(self):
        plt.close()

    def show_plot(self):
        plt.show()

    def plot_hidden(self, hidden, position, plot_type, lim=None):
        p1 = hidden[:, 0]
        p2 = hidden[:, 1]

        if lim is not None:
            plt.xlim(lim[0][0], lim[0][1])
            plt.ylim(lim[1][0], lim[1][1])

        if plot_type in ['self_position']:
            t1 = position[:, 0]
            t2 = position[:, 1]
            t_idx1 = (t1 + self.field_size).astype(numpy.int)
            t_idx2 = (t2 + self.field_size).astype(numpy.int)
            color = self.colormap[t_idx1, t_idx2]
            d = self.ax.scatter(p1, p2, c=color, alpha=0.9, zorder=0)
        elif plot_type in ['other_position']:
            t1 = position[:, 0]
            t2 = position[:, 1]
            t_idx1 = (t1 + self.field_size).astype(numpy.int)
            t_idx2 = (t2 + self.field_size).astype(numpy.int)
            color = self.colormap[t_idx1, t_idx2]
            d = self.ax.scatter(p1, p2, c=color, alpha=0.9, zorder=0)
        elif plot_type == 'point':
            d = self.ax.scatter(p1, p2, c='w', edgecolor='k', s=200, zorder=0)
        else:
            assert (False)
        return d
