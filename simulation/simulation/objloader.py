import os

from .geometry import Material, Model, ModelGroup
from .shader import Shader


def replace_newlinecode(line):
    return line.replace('\n', '').replace('\r', '')


def detect_line(lines, target):
    for i, line in enumerate(lines):
        if line.startswith(target):
            group_name = line[len(target) + 1:]
            yield (i, group_name)
    yield (None, None)


class ObjLoader(object):
    def __init__(self):
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + '/shader/'
        self.shaders = {
            'color':
            Shader(
                vs_program=shader_dir + '/color_vs.glsl',
                fs_program=shader_dir + '/color_fs.glsl',
            ),
            'texture':
            Shader(
                vs_program=shader_dir + '/texture_vs.glsl',
                fs_program=shader_dir + '/texture_fs.glsl',
            ),
        }

    def load_obj(self, obj_file):
        self.dir = os.path.dirname(os.path.abspath(obj_file)) + '/'
        with open(obj_file, 'r') as f:
            lines = map(replace_newlinecode, list(f.readlines()))

        mtl_file = os.path.splitext(obj_file)[0] + '.mtl'
        mtl_file = mtl_file if os.path.isfile(mtl_file) else None

        with open(mtl_file, 'r') as f:
            mtl_lines = map(replace_newlinecode, list(f.readlines()))

        mtls = self.load_material(mtl_lines)
        models = self.load(lines, mtls)

        return ModelGroup(models)

    def load_material(self, lines):

        mtls = {}
        for line in lines:
            if line.startswith('newmtl '):
                _, name = line.split(' ')
                current_mtl = Material()
                current_mtl.set_name(name)
                mtls[name] = current_mtl
                continue

            if line.startswith('Ns '):
                _, Ns = line.split(' ')
                current_mtl.set_Ns(float(Ns))
            elif line.startswith('Ka '):
                _, r, g, b = line.split(' ')
                current_mtl.set_Ka(tuple(map(float, [r, g, b])))
            elif line.startswith('Kd '):
                _, r, g, b = line.split(' ')
                current_mtl.set_Kd(tuple(map(float, [r, g, b])))
            elif line.startswith('Ks '):
                _, r, g, b = line.split(' ')
                current_mtl.set_Ks(tuple(map(float, [r, g, b])))
            elif line.startswith('Ni '):
                _, Ni = line.split(' ')
                current_mtl.set_Ni(float(Ni))
            elif line.startswith('d '):
                _, d = line.split(' ')
                current_mtl.set_d(float(d))
            elif line.startswith('illum '):
                _, illum = line.split(' ')
                current_mtl.set_illum(float(illum))
            elif line.startswith('map_Kd '):
                _, map_Kd = line.split(' ')
                current_mtl.set_map_Kd(self.dir + map_Kd)

        return mtls

    def load(self, lines, mtls):
        v = []
        vt = []
        vn = []

        models = []
        for line in lines:
            if line.startswith('o '):
                _, object_name = line.split(' ')
                model = Model(object_name)
                models.append(model)
                continue

            if line.startswith('v '):
                _, x, y, z = line.split(' ')
                v.append(list(map(float, [x, y, z])))
            elif line.startswith('vt '):
                _, x, y = line.split(' ')[:3]
                vt.append(list(map(float, [x, y])))
            elif line.startswith('vn '):
                _, x, y, z = line.split(' ')
                vn.append(list(map(float, [x, y, z])))
            elif line.startswith('f '):
                for xyz in line.split(' ')[1:]:
                    for i, value in enumerate(xyz.split('/')):
                        if len(value) == 0:
                            continue
                        index = int(value) - 1
                        if i == 0:
                            model.append_vertices(v[index])
                        elif i == 1:
                            model.append_uvs(vt[index])
                        elif i == 2:
                            model.append_normals(vn[index])
            elif line.startswith('usemtl '):
                _, usemtl = line.split(' ')
                model.set_usemtl(usemtl)
                model.set_material(mtls[usemtl])
                model.set_shader(
                    self.shaders['texture']
                    if mtls[usemtl].has_map_Kd() else self.shaders['color']
                )

        for model in models:
            model.initialize()

        return models
