import glfw
import OpenGL.GL as gl


class GLWorld(object):
    def gl_init(self, width, height):
        assert (glfw.init())

        if not (self.display):
            glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(width, height, 'Window', None, None)
        if self.display:
            glfw.set_key_callback(self.window, self.key_callback)
        assert (not (self.window is None))
        glfw.make_context_current(self.window)

        fogColor = [0.0, 0.0, 0.0, 1.0]

        gl.glClearColor(0.5, 0.65, 0.85, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        gl.glFogi(gl.GL_FOG_MODE, gl.GL_LINEAR)
        gl.glFogfv(gl.GL_FOG_COLOR, fogColor)
        gl.glFogf(gl.GL_FOG_DENSITY, 0.99)
        gl.glHint(gl.GL_FOG_HINT, gl.GL_DONT_CARE)
        gl.glFogf(gl.GL_FOG_START, 1.0)
        gl.glFogf(gl.GL_FOG_END, 5.0)
        gl.glEnable(gl.GL_FOG)

    def close(self):
        glfw.Terminate()

    def resize_window(self, width, height):
        glfw.set_window_size(self.window, width, height)
