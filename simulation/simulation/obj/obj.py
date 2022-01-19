from .point import Point

from .. import glm


class Obj(Point):
    def __init__(
            self,
            model,
            translation=(0, 0, 0),
            rotation=(0, 1, 0, 0),
            scale=(1.0, 1.0, 1.0),
            visible=True
    ):
        self.model = model
        self.visible = visible
        self.scale = list(scale)

        super(Obj, self).__init__(translation=translation, rotation=rotation)

        self.__update_m_matrix()

    def toggle_visible(self):
        self.visible = not (self.visible)

    def set_visible(self, visible):
        self.visible = visible

    def __update_m_matrix(self):
        self.__m_matrix = self.__calc_m_matrix()

    def __calc_m_matrix(self):
        return glm.scale(*(self.scale)) * \
            glm.rotation(*(self.rotation)) * \
            glm.translation(*(self.translation))

    def set_scale(self, sx, sy, sz):
        self.scale[0] = sx
        self.scale[1] = sy
        self.scale[2] = sz

    def draw(self, p_matrix, v_matrix, updated=False):
        if updated:
            self.__update_m_matrix()

        if self.visible:
            self.model.draw(p_matrix, v_matrix, self.__m_matrix)
