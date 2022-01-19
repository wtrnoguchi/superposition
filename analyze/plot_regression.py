import argparse
import subprocess
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--fontsize', type=int, default=24)
    parser.add_argument('--ticks_fontsize', type=int, default=18)
    args = parser.parse_args()

    savedir_root = 'regression/{mode}/'.format(**vars(args))
    plotdir = savedir_root + "plot/"
    subprocess.check_output(['mkdir', '-p', plotdir])

    mses = {}
    for target_so in ["self", "other"]:
        for layer_so in ["self", "other"]:
            _mses = []

            for epoch in range(0, 200+1):
                f_name = "{:05d}_layer-{}_terget-{}.txt".format(
                    epoch, layer_so, target_so)
                mse = numpy.loadtxt(savedir_root + f_name)
                _mses.append(mse)

            mses[(target_so, layer_so)] = numpy.array(_mses)

    for target_so in ["self", "other"]:
        for layer_so in ["self", "other"]:
            # label = target_so + "_" + layer_so
            label = layer_so + "_" + target_so
            c = "r" if layer_so == "self" else "b"
            linestyle = "solid" if layer_so == target_so else "dashed"
            plt.plot(mses[(target_so, layer_so)],
                     label=label, c=c, linestyle=linestyle)

    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(plotdir + "1d.png")
    plt.savefig(plotdir + "1d.eps")
    plt.savefig(plotdir + "1d.pdf")
    plt.close()

    plt.plot(mses[("self", "self")], mses[("other", "self")],
             label="self", c="r", markevery=[0], marker="o")
    plt.plot(mses[("self", "other")], mses[("other", "other")],
             label="other", c="b", markevery=[0], marker="o")
    plt.xlabel("MSE for self location", fontsize=args.fontsize)
    plt.ylabel("MSE for other location", fontsize=args.fontsize)
    plt.xticks(fontsize=args.ticks_fontsize)
    plt.yticks(fontsize=args.ticks_fontsize)
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect(1.0/ax.get_data_ratio())
    plt.savefig(plotdir + "2d.png")
    plt.savefig(plotdir + "2d.eps")
    plt.savefig(plotdir + "2d.pdf")
