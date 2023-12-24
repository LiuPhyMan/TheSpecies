# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

import re
from math import log

import numpy as np
from myconst import relM2absM, eV2K, eV2J, K2eV, hbar, k as kB, allSR
from pandas import read_csv
from scipy.interpolate import interp1d

from .line import AtomLines

__all__ = ["spec_df", "SpecieWithLevel", "SpecieWithQint", "spc_dict"]

spec_df = read_csv(__path__[0] + r"/specie_data.csv", sep=",", header=None, index_col=0,
                   names=["relM", "Zc", "Hf", "ionE", "polar", "elems"], comment="#")


class AbsSpecie(object):
    __slots__ = ["spc_str", "relM", "absM", "Zc", "Hf", "ionE", "polar", "elems", "lines"]

    def __init__(self, *, spc_str: str) -> None:
        assert spc_str in spec_df.index, spc_str
        self.spc_str = spc_str
        self.relM = spec_df.loc[spc_str, "relM"]
        self.absM = self.relM*relM2absM
        self.Zc = spec_df.loc[spc_str, "Zc"]
        self.Hf = spec_df.loc[spc_str, "Hf"]
        self.ionE = float(spec_df.loc[spc_str, "ionE"]) if not spec_df.loc[
                                                                   spc_str, "ionE"].strip() == "" else 0
        self.polar = float(spec_df.loc[spc_str, "polar"]) if not spec_df.loc[
                                                                     spc_str, "polar"].strip() == "" else 0
        self.elems = {_[0]: int(_[1]) for _ in re.findall(r"([a-zA-Z]+)-(\d+)",
                                                          spec_df.loc[spc_str, "elems"])}

    @property
    def type(self):
        if self.Zc == 0:
            return "neu"
        if self.Zc > 0:
            return "ion"
        if self.Zc < 0:
            return "negChrg"

    def get_nElem(self, element: str):
        if element == "e":
            return -self.Zc
        else:
            return self.elems.get(element, 0)

    def qint(self, T_K):
        pass

    def set_emiss_lines(self, *, lineFile):
        self.lines = AtomLines(lineFile=lineFile)

    def norm_j_bb(self, *, T_K: float, wvlnm_rng: tuple) -> float:
        # nu_range = (light_c / wv_range[1], light_c / wv_range[0])
        df = self.lines.df_filter_by_wv_range(wvlnm_rng=wvlnm_rng)
        if df.index.size == 0:
            return 0
        tmp = np.dot(df["wAg"], np.exp(-df["Ek_eV"]/(T_K*K2eV)))
        return hbar/self.qint(T_K=T_K)*tmp/allSR


class _Electron(AbsSpecie):

    def __init__(self) -> None:
        super().__init__(spc_str="e")
        self.elems = {"e": 1}

    def norm_j_bb(self, *, T_K: float, wvlnm_rng) -> float:
        return 0

    def qint(self, T_K):
        return 2

    def lnqint(self, T_K):
        return log(2)

    def rdcd_mu0(self, *, T_K):
        r""" \mu(T) / kT  or \mu(T)/RT"""
        return 3.66487052 - 1.5*log(self.relM) - 2.5*log(T_K) - log(2)

    def get_h(self, *, T_K):
        return 2.5*kB*T_K


class SpecieWithLevel(AbsSpecie):

    def __init__(self, *, spc_str: str) -> None:
        super().__init__(spc_str=spc_str)

    def set_gE(self, *, gE_file: str):
        _data = np.loadtxt(gE_file, delimiter=",")
        self.g = _data[:, 0]
        self.E = _data[:, 1]

    def qint(self, T_K: float):
        return np.dot(self.g, np.exp(-self.E*eV2K/T_K))

    def lnqint(self, T_K: float):
        return log(self.qint(T_K))

    def dlnqintdT(self, T_K: float):
        T_eV = T_K*K2eV
        return (T_K*self.qint(T_K))**(-1)*np.dot(self.g,
                                                 self.E/T_eV*np.exp(-self.E/T_eV))

    def dlnqintdT_2(self, T_K: float):
        T_eV = T_K*K2eV
        temp = np.dot(self.g*self.E/T_eV*
                      (-2 + self.E/T_eV), np.exp(-self.E/T_eV))
        return (T_K*self.qint(T_K))**(-1)/T_K*temp

    def rdcd_mu0(self, *, T_K):
        return 3.66487052 - 1.5*log(self.relM) - 2.5*log(T_K) - log(self.qint(T_K)) \
            + self.Hf*eV2K/T_K

    def get_h(self, *, T_K: float):
        return 2.5*kB*T_K + kB*T_K**2*self.dlnqintdT(T_K) + self.Hf*eV2J

    def gnd_frac(self, T_K: float):
        return self.g[0]/self.qint(T_K)


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
        return float(interp1d(self._data[:, 0], self._data[:, 3])(T_K)/T_K)

    def dlnqintdT_2(self, T_K: float):
        dlnqdT = self.dlnqintdT(T_K)
        return float((interp1d(self._data[:, 0],
                               self._data[:, 4])(T_K) - 2*T_K*dlnqdT)/T_K**2)

    def rdcd_mu0(self, *, T_K: float):
        return 3.66487052 - 1.5*log(self.relM) - 2.5*log(T_K) - log(self.qint(T_K)) + \
            self.Hf*eV2K/T_K

    def get_h(self, *, T_K: float):
        return 2.5*kB*T_K + kB*T_K**2*self.dlnqintdT(T_K) + self.Hf*eV2J

    # def cp(self, T_K: float):
    #     return 2.5 + 2 * T_K * self.dlnqintdT(T_K) + T_K**2 * self.dlnqintdT_2(T_K)


# =============================================================================================== #
with open(__path__[0] + r"/spec_info.txt") as f:
    spec_info = f.readlines()

spc_dict = dict()
spc_dict["e"] = _Electron()
for _str in spec_info[:10]:
    _key, _type, _path = [_.strip() for _ in _str.split(",")]
    if _type == "level":
        spc_dict[_key] = SpecieWithLevel(spc_str=_key)
        spc_dict[_key].set_gE(gE_file=f"{__path__[0]}/data/{_path}")
    elif _type == "qint":
        spc_dict[_key] = SpecieWithQint(spc_str=_key)
        spc_dict[_key].set_qint(qint_file=f"{__path__[0]}/data/{_path}")
    else:
        raise Exception(f"The type {_type} is error.")

spc_dict["Ar"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Ar/Ar I.txt")
spc_dict["Ar_1p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Ar/Ar II.txt")
spc_dict["Ar_2p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Ar/Ar III.txt")
spc_dict["Ar_3p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Ar/Ar IV.txt")
spc_dict["Ar_4p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Ar/Ar V.txt")
spc_dict["Xe"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Xe/Xe I.txt")
spc_dict["Xe_1p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Xe/Xe II.txt")
spc_dict["Xe_2p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Xe/Xe III.txt")
spc_dict["Xe_3p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Xe/Xe IV.txt")
spc_dict["Xe_4p"].set_emiss_lines(lineFile=f"{__path__[0]}/data/line/Xe/Xe V.txt")
# spc_dict = dict()
# for _str in _spc_info:
#     if _str[0] == "gE":
#         spc_dict[_str[1]] = Specie
