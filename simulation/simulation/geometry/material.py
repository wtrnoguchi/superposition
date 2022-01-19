class Material(object):
    def __init__(
            self,
            name=None,
            Ns=None,
            Ka=None,
            Kd=None,
            Ks=None,
            Ni=None,
            d=None,
            illum=None,
            map_Kd=None,
    ):

        self.__name = name
        self.__Ns = Ns
        self.__Ka = Ka
        self.__Kd = Kd
        self.__Ks = Ks
        self.__Ni = Ni
        self.__d = d
        self.__illum = illum
        self.__map_Kd = map_Kd

    def get_name(self):
        return self.__name

    def get_Ns(self):
        return self.__Ns

    def get_Ka(self):
        return self.__Ka

    def get_Kd(self):
        return self.__Kd

    def get_Ks(self):
        return self.__Ks

    def get_Ni(self):
        return self.__Ni

    def get_d(self):
        return self.__d

    def get_illum(self):
        return self.__illum

    def get_map_Kd(self):
        return self.__map_Kd

    def set_name(self, name):
        self.__name = name

    def set_Ns(self, Ns):
        self.__Ns = Ns

    def set_Ka(self, Ka):
        self.__Ka = Ka

    def set_Kd(self, Kd):
        self.__Kd = Kd

    def set_Ks(self, Ks):
        self.__Ks = Ks

    def set_Ni(self, Ni):
        self.__Ni = Ni

    def set_d(self, d):
        self.__d = d

    def set_illum(self, illum):
        self.__illum = illum

    def set_map_Kd(self, map_Kd):
        self.__map_Kd = map_Kd

    def has_map_Kd(self):
        return self.__map_Kd is not None
