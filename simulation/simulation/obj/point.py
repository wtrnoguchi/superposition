class Point(object):
    def __init__(self, translation=(0, 0, 0), rotation=(0, 1, 0, 0)):

        self.translation = list(translation)
        self.rotation = list(rotation)

    def set_translation(self, x, y, z):
        self.translation[0] = x
        self.translation[1] = y
        self.translation[2] = z

    def set_xz(self, x, z):
        self.translation[0] = x
        self.translation[2] = z

    def set_rotation(self, deg, rx, ry, rz):
        self.rotation[0] = deg
        self.rotation[1] = rx
        self.rotation[2] = ry
        self.rotation[3] = rz

    def set_degree(self, degree):
        self.rotation[0] = degree

    def add_x(self, dx):
        self.translation[0] += dx

    def add_y(self, dy):
        self.translation[1] += dy

    def add_z(self, dz):
        self.translation[2] += dz

    def add_deg(self, ddeg):
        self.rotation[0] += ddeg
