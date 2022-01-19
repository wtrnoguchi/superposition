import ctypes

import numpy
from OpenGL import GL as gl
from PIL import Image


class ModelBase(object):
    def _set_mesh(self):
        pass

    def _set_vao(self):
        pass

    def draw(self, p_matrix, v_matrix, m_matrix):
        pass


class ModelGroup(ModelBase):
    def __init__(self, models=[]):
        self.models = models

    def add_model(self, model):
        self.models.append(model)

    def draw(self, p_matrix, v_matrix, m_matrix):
        for model in self.models:
            model.draw(p_matrix, v_matrix, m_matrix)


class Model(ModelBase):
    def __init__(self, name):
        self.name = name
        self.init_vertices()
        super(Model, self).__init__()

    def initialize(self):
        self._set_mesh()
        self._set_vao()

    def init_vertices(self):
        self.vertices = []
        self.uvs = []
        self.normals = []

    def append_vertices(self, vertices):
        self.vertices += vertices

    def append_uvs(self, uvs):
        self.uvs += uvs

    def append_normals(self, normals):
        self.normals += normals

    def set_usemtl(self, usemtl):
        self.usemtl = usemtl

    def set_material(self, material):
        self.material = material

    def _set_mesh(self):
        self.n_vertices = len(self.vertices)

        ambient = [self.material.get_Ka()] * self.n_vertices
        self.ambient = numpy.array(ambient, numpy.float32).reshape(-1)

        diffuse = [self.material.get_Kd()] * self.n_vertices
        self.diffuse = numpy.array(diffuse, numpy.float32).reshape(-1)

        specular = [self.material.get_Ks()] * self.n_vertices
        self.specular = numpy.array(specular, numpy.float32).reshape(-1)

        specular_intensity = [self.material.get_Ns()] * self.n_vertices
        self.specular_intensity = numpy.array(
            specular_intensity, numpy.float32
        ).reshape(-1)

        self.vertex = numpy.array(self.vertices, numpy.float32)
        self.normal = numpy.array(
            self.normals, numpy.float32
        ) if self.normals else None
        self.uv = numpy.array(self.uvs, numpy.float32) if self.uvs else None

    def draw(self, p_matrix, v_matrix, m_matrix):
        gl.glUseProgram(self.shader.program)
        self.shader.set_projection_matrix(p_matrix)
        self.shader.set_view_matrix(v_matrix)
        self.shader.set_model_matrix(m_matrix)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glBindSampler(0, self.sampler)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.n_vertices)
        gl.glBindVertexArray(0)

    def _set_vao(self):
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        if self.material.has_map_Kd():
            map_Kd = self.material.get_map_Kd()
            img = Image.open(map_Kd)
            img_data = numpy.array(list(img.getdata()), numpy.uint8)[:, :3]
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img.size[0], img.size[1], 0,
                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data
            )

        self.sampler = gl.glGenSamplers(1)
        gl.glSamplerParameteri(self.sampler, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glSamplerParameteri(self.sampler, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glSamplerParameteri(
            self.sampler, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR
        )
        gl.glSamplerParameteri(
            self.sampler, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.assign_vertex(0, self.vertex, 3)
        if self.uv is not None:
            self.assign_vertex(1, self.uv, 2)
        if self.normal is not None:
            self.assign_vertex(2, self.normal, 3)
        self.assign_vertex(3, self.ambient, 3)
        self.assign_vertex(4, self.diffuse, 3)
        self.assign_vertex(5, self.specular, 3)
        self.assign_vertex(6, self.specular_intensity, 1)

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)

    def assign_vertex(self, layout_idx, vertices, unit_size):
        gl.glEnableVertexAttribArray(layout_idx)
        buf = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, 4 * len(vertices), vertices, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(
            layout_idx, unit_size, gl.GL_FLOAT, gl.GL_FALSE, 0,
            ctypes.c_void_p(0)
        )

    def set_shader(self, shader):
        self.shader = shader
