import OpenGL.GL as gl
from OpenGL.GL import shaders


def _read_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return content


class Shader(object):
    def __init__(self, vs_program, fs_program):
        self.vs = shaders.compileShader(
            _read_file(vs_program), gl.GL_VERTEX_SHADER
        )
        self.fs = shaders.compileShader(
            _read_file(fs_program), gl.GL_FRAGMENT_SHADER
        )
        self.program = shaders.compileProgram(self.vs, self.fs)

        self.p_matrix = gl.glGetUniformLocation(self.program, 'projection')
        self.v_matrix = gl.glGetUniformLocation(self.program, "view")
        self.m_matrix = gl.glGetUniformLocation(self.program, "model")
        self.texture = gl.glGetUniformLocation(self.program, "uTexture")

    def set_projection_matrix(self, p_matrix):
        gl.glUniformMatrix4fv(self.p_matrix, 1, gl.GL_FALSE, p_matrix)

    def set_view_matrix(self, v_matrix):
        gl.glUniformMatrix4fv(self.v_matrix, 1, gl.GL_FALSE, v_matrix)

    def set_model_matrix(self, m_matrix):
        gl.glUniformMatrix4fv(self.m_matrix, 1, gl.GL_FALSE, m_matrix)
