# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

from math import log
import re
import numpy as np
from scipy.interpolate import interp1d
from pandas import read_csv
from myconst import relM2absM, eV2K, K2J, K2eV

__all__ = ["spec_df", "SpecieWithLevel", "SpecieWithQint"]

spec_df = read_csv(__path__[0] + r"/specie_data.csv", sep=",", header=None, index_col=0, 
                   names=["relM", "Zc", "Hf", "elems"], comment="#")


class AbsSpecie(object):

    def __init__(self, *, spc_str:str) -> None:
        assert spc_str in spec_df.index, spc_str
        self.relM = spec_df.loc[spc_str, "relM"]
        self.absM = self.relM * relM2absM
        self.Zc = spec_df.loc[spc_str, "Zc"]
        self.Hf = spec_df.loc[spc_str, "Hf"]
        self.elems = {_[0]:int(_[1]) for _ in re.findall(r"([a-zA-Z]+)-(\d+)", 
                                                         spec_df.loc[spc_str, "elems"])}

    def get_nElem(self, element: str):
        return self.elems.get(element, 0)

    def qint(self, T_K):
        pass

class SpecieWithLevel(AbsSpecie):

    def __init__(self, *, spc_str: str) -> None:
        super().__init__(spc_str=spc_str)

    def set_gE(self, *, gE_file: str):
        _data = np.loadtxt(gE_file, delimiter=",")
        self.g = _data[:, 0]
        self.E = _data[:, 1]

    def qint(self, T_K: float):
        return np.dot(self.g, np.exp(-self.E * eV2K / T_K))

    def lnqint(self, T_K: float):
        return log(self.qint(T_K))

    def dlnqintdT(self, T_K: float):
        T_eV = T_K * K2eV
        return (T_K * self.qint(T_K)) ** (-1) * np.dot(self.g,
                                                       self.E / T_eV * np.exp(-self.E / T_eV))

    def dlnqintdT_2(self, T_K: float):
        T_eV = T_K * K2eV
        temp = np.dot(self.g * self.E / T_eV * (-2 + self.E / T_eV), np.exp(-self.E / T_eV))
        return (T_K * self.qint(T_K)) ** (-1) / T_K * temp

    def rdcd_mu0(self, *, T_K):
        return -88.8291123098254 - 1.5 * log(self.absM) - 2.5 * log(T_K) - log(self.qint(T_K)) \
            + self.Hf / (T_K * K2J)



class SpecieWithQint(AbsSpecie):

    def __init__(self, *, spc_str: str) -> None:
        super().__init__(spc_str=spc_str)

    def set_qint(self, *, qint_file: str):
        self._data = np.loadtxt(qint_file, delimiter=",")

    def qint(self, T_K: float):
        return float(interp1d(self._data[:, 0], self._data[:, 1])(T_K))

    def lnqint(self, T_K: float):
        return float(interp1d(self._data[:, 0], self._data[:, 2])(T_K))

    def dlnqintdT(self, T_K: float):
        return float(interp1d(self._data[:, 0], self._data[:, 3])(T_K) / T_K)

    def dlnqintdT_2(self, T_K: float):
        dlnqdT = self.dlnqintdT(T_K)
        return float((interp1d(self._data[:, 0],
                               self._data[:, 4])(T_K) - 2 * T_K * dlnqdT) / T_K ** 2)

    def rdcd_mu0(self, *, T_K: float):
        return -88.8291123098254 - 1.5 * log(self.absM) - 2.5 * log(T_K) - log(self.qint(T_K)) + \
            self.Hf / (T_K * K2J)

with open(__path__[0] + r"/spec_info.txt") as f:
    spec_info = f.readlines()

spc_dict = dict()
for _str in spec_info[:5]:
    _key, _type, _path = [_.strip() for _ in _str.split(",")]
    if _type == "level":
        spc_dict[_key] = SpecieWithLevel(spc_str=_key)
        spc_dict[_key].set_gE(gE_file=f"{__path__[0]}/{_path}")
    elif _type == "qint":
        spc_dict[_key] = SpecieWithQint(spc_str=_key)
        spc_dict[_key].set_qint(qint_file=f"{__path__[0]}/{_path}")
    else:
        raise Exception(f"The type is error.")





# spc_dict = dict()
# for _str in _spc_info:
#     if _str[0] == "gE":
#         spc_dict[_str[1]] = Specie

