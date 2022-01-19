import itertools
import math

from OpenGL import GL as gl

from . import glm


class Camera(object):
    def __init__(
            self,
            config,
            body,
    ):

        for k, v in config.items():
            setattr(self, k, v)

        self.set_body(body)

        self.aspect = float(self.width) / self.height

        self.hfov_seg = self.hfov / self.n_camera

        self.camera_dirs = [
            self.hfov_seg / 2. - self.hfov / 2. + i * self.hfov_seg
            for i in range(self.n_camera)
        ]

        self.projection = 'perspective'

    def _draw(self, draw_func, rotate=0):
        draw_func(self.get_p_matrix(), self.gen_v_matrix(rotate))

    def set_body(self, body):
        self.__body = body

    def read_pixel(self):
        gl.glReadBuffer(gl.GL_BACK)
        dataBuffer = gl.glReadPixels(
            0,
            0,
            self.width,
            self.height,
            gl.GL_BGR,
            gl.GL_FLOAT,
        )

        return dataBuffer.reshape(self.height, self.width, 3).transpose(
            0, 1, 2)

    def draw(self, draw_func):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        uw = int(self.width / self.n_camera)
        for i, r in enumerate(reversed(self.camera_dirs)):
            wl = uw * i
            gl.glViewport(wl, 0, uw, self.height)

            self._draw(draw_func, r)

    def __update_p_matrix(self):
        if self.projection == 'perspective':

            tany = math.tan(math.pi * self.vfov / 180 / 2)
            tanx = math.tan(math.pi * self.hfov_seg / 180 / 2)
            aspect = tanx / tany

            self.__p_matrix = glm.perspective(self.vfov, aspect, self.near,
                                              self.far)

        elif self.projection == 'orthogonal':
            self.__p_matrix = glm.orthogonal(
                self.overview_area[0][1], self.overview_area[0][0],
                self.overview_area[1][1], self.overview_area[1][0], self.near,
                self.far)
        else:
            assert (False)

    def set_projection(self, projection):
        self.projection = projection
        self.__update_p_matrix()

    def __v_matrix(self, rotate=0):
        x, y, z = self.__body.translation
        deg, rx, ry, rz = self.__body.rotation
        return glm.translation(-x, -y, -z) * glm.rotation(
            -(deg + rotate), rx, ry, rz)

    def get_p_matrix(self):
        return self.__p_matrix

    def gen_v_matrix(self, rotate=0):
        return self.__v_matrix(rotate)
