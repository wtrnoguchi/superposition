import itertools
import os

import glfw

from .camera import Camera
from .common_world import GLWorld
from .obj import Obj, Point
from .objloader import ObjLoader


class World(GLWorld):
    ANGLE_OFFSET = 90

    def __init__(self, config, display=True):
        self.config = config
        self.display = display

    def off_display(self):
        self.display = False

    def init(self):

        self.gl_init(self.config.camera.agent.width,
                     self.config.camera.agent.height)

        model_dir = os.path.dirname(
            os.path.abspath(__file__)) + '/resource/model/'
        self.obj_loader = ObjLoader()

        self.objects = []

        self_model = self.obj_loader.load_obj(model_dir +
                                              self.config.model.self)

        self.self = Obj(
            model=self_model,
            translation=(0, self.config.eye_level, 0),
            rotation=(0, 0, 1, 0),
            visible=True,
        )

        self.objects.append(self.self)

        field_model = self.obj_loader.load_obj(model_dir +
                                               self.config.model.field)

        self.field = Obj(
            model=field_model,
            translation=(0, 0, 0),
            rotation=(0, 0, 1, 0),
            visible=True,
        )
        self.objects.append(self.field)

        self.other_model = self.obj_loader.load_obj(model_dir +
                                                    self.config.model.other)

        self.other = Obj(
            model=self.other_model,
            translation=(0, self.config.eye_level, 0),
            rotation=(0, 0, 1, 0),
            visible=True,
        )
        self.objects.append(self.other)

        self.boundary = (
            (self.config.xmin, self.config.xmax),
            (self.config.ymin, self.config.ymax),
        )

        self.__init_camera()

    def __init_camera(self):
        self_camera = Camera(
            config=self.config.camera.agent,
            body=self.self,
        )

        other_camera = Camera(
            config=self.config.camera.agent,
            body=self.other,
        )

        overview_point = Point(
            translation=(0, 10, 0),
            rotation=(-90, 1, 0, 0),
        )

        overview_camera = Camera(
            self.config.camera.overview, body=overview_point)
        self.cameras = {
            'self': self_camera,
            'other': other_camera,
            'overview': overview_camera
        }

        self.__camera_cycle = itertools.cycle(self.cameras.keys())
        self.set_camera('self')

    def set_camera(self, key):
        self.current_camera_id = key
        self.update_camera()

    def cycle_camera(self):
        self.current_camera_id = next(self.__camera_cycle)
        self.update_camera()

    def set_visible(self, visible_self, visible_other):
        self.self.set_visible(visible_self)
        self.other.set_visible(visible_other)

    def update_camera(self):
        if self.current_camera_id == 'self':
            self.camera = self.cameras[self.current_camera_id]
            self.set_visible(False, True)
            self.camera.set_projection('perspective')
        elif self.current_camera_id == 'other':
            self.camera = self.cameras[self.current_camera_id]
            self.set_visible(True, False)
            self.camera.set_projection('perspective')
        elif self.current_camera_id == 'overview':
            self.camera = self.cameras[self.current_camera_id]
            self.set_visible(True, True)
            self.camera.set_projection('orthogonal')
        else:
            assert (False)
        self.resize_window(self.camera.width, self.camera.height)

    def _draw(self, p_matrix, v_matrix):

        for obj in self.objects:
            obj.draw(p_matrix, v_matrix, updated=True)

    def draw(self, s, o):
        self.self.set_xz(s.p[0], s.p[1])
        self.self.set_degree(-self.ANGLE_OFFSET)

        self.other.set_xz(o.p[0], o.p[1])
        self.other.set_degree(-self.ANGLE_OFFSET)

        self.camera.draw(self._draw)

        if self.display:
            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def get_boundary(self):
        return self.boundary

    def capture(self):
        vision = self.camera.read_pixel()

        return vision[::-1].copy()

    def key_callback(self, window, key, scancode, action, mods):
        if action in [glfw.PRESS]:
            if key == glfw.KEY_ENTER:
                self.is_run = not (self.is_run)
            elif key == glfw.KEY_ESCAPE:
                self.close()
            elif key == glfw.KEY_C:
                self.cycle_camera()
