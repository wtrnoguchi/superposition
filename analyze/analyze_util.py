import numpy

import cv2


def make2Dcolormap(
        colors=(
            (1, 1, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        ), size=20):
    ######################
    colormap = numpy.zeros((2, 2, 3))
    colormap[1, 1] = colors[0]
    colormap[0, 1] = colors[1]
    colormap[0, 0] = colors[2]
    colormap[1, 0] = colors[3]
    size = size + 1
    colormap = cv2.resize(colormap, (size, size))
    colormap = numpy.clip(colormap, 0, 1)
    return colormap


def flat_combine(lst):
    return numpy.concatenate([x.reshape(-1, x.shape[-1]) for x in lst])
