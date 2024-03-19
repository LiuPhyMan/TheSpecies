# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

import re
from math import log
from pathlib import Path

import numpy as np
from myconst import relM2absM, eV2K, eV2J, K2eV, kB
from pandas import read_csv
from scipy.interpolate import interp1d

parentPath = Path(__file__).parent
spec_df = read_csv(str(parentPath/"specie_data.csv"), sep=",", header=None,
                   names=["relM", "Zc", "Hf", "ionE", "polar", "elems"],
                   index_col=0, comment="#")


class _AbsSpecie(object):
  __slots__ = ["spc_str", "relM", "absM", "Zc", "Hf", "_ionE", "polar",
               "elems", "lines"]

  def __init__(self, *, spc_str: str) -> None:
    assert spc_str in spec_df.index, spc_str
    _spc = spec_df.loc[spc_str]
    self.spc_str = spc_str
    self.relM = _spc["relM"]
    self.absM = self.relM*relM2absM
    self.Zc = _spc["Zc"]
    self.Hf = _spc["Hf"]
    self._ionE = None if _spc["ionE"].strip()=="" else float(_spc["ionE"])
    self.polar = None if _spc["polar"].strip()=="" else float(_spc["polar"])
    self.elems = {_[0]: int(_[1])
                  for _ in re.findall(r"([a-zA-Z]+)-(\d+)", _spc["elems"])}

  def isNeu(self):
    return True if self.Zc==0 else False

  def isPosIon(self):
    return True if self.Zc > 0 else False

  def isNegIon(self):
    return True if self.Zc < 0 else False

  def get_nElem(self, element: str):
    if element=="e":
      return -self.Zc
    else:
      return self.elems.get(element, 0)

  def qint(self, T_K):
    pass

  def lnqint(self, T_k):
    pass

  def rdcd_mu0(self, *, T_trans, T_int):
    r""" \mu(T) / kT  or \mu(T)/RT"""
    # TODO why Hf * eV2K/T_int
    return 3.66487052 - 1.5*log(self.relM) - 2.5*log(T_trans) - \
      self.lnqint(T_int) + self.Hf*eV2K/T_trans

  # def chem_ptn(self, *, T_trans, T_int, lnN, V):
  #   r"""chemical potential in unit of K."""
  #   ln_debrog_wvlth = -20.16604046 - 0.5*log(self.relM)-0.5*log(T_trans)
  #   tmp = -log(V)+3*ln_debrog_wvlth - self.lnqint(T_int) + \
  #         lnN + self.Hf*eV2K/T_trans
  #   return tmp * T_trans


class _Electron(_AbsSpecie):

  def __init__(self) -> None:
    super().__init__(spc_str="e")
    self.elems = {"e": 1}

  def qint(self, T_K):
    return 2

  def lnqint(self, T_K):
    return log(2)

  def get_h(self, *, T_K):
    return 2.5*kB*T_K


class SpecieWithLevel(_AbsSpecie):

  def __init__(self, *, spc_str: str) -> None:
    super().__init__(spc_str=spc_str)

  def set_gE(self, *, gE_file: Path):
    _data = np.loadtxt(str(gE_file), delimiter=",")
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
    temp = np.dot(self.g*self.E/T_eV*(-2 + self.E/T_eV),
                  np.exp(-self.E/T_eV))
    return (T_K*self.qint(T_K))**(-1)/T_K*temp

  def get_h(self, *, T_K: float):
    return 2.5*kB*T_K + kB*T_K**2*self.dlnqintdT(T_K) + self.Hf*eV2J

  def gnd_frac(self, T_K: float):
    return self.g[0]/self.qint(T_K)


class SpecieWithQint(_AbsSpecie):

  def __init__(self, *, spc_str: str) -> None:
    super().__init__(spc_str=spc_str)

  def set_qint(self, *, qint_file: Path):
    self._data = np.loadtxt(str(qint_file), delimiter=",")

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

  def get_h(self, *, T_K: float):
    return 2.5*kB*T_K + kB*T_K**2*self.dlnqintdT(T_K) + self.Hf*eV2J


# =========================================================================== #
with (parentPath/'spec_info.txt').open() as f:
  spec_info = [_l for _l in f.readlines() if not _l.startswith('#')]

spc_dict = dict()
spc_dict["e"] = _Electron()
for _str in spec_info:  # top 10 lines.
  _key, _type, _path = [_.strip() for _ in _str.split(",")]
  if _type=="level":
    spc_dict[_key] = SpecieWithLevel(spc_str=_key)
    spc_dict[_key].set_gE(gE_file=parentPath/'data'/'level'/_path)
  elif _type=="qint":
    spc_dict[_key] = SpecieWithQint(spc_str=_key)
    spc_dict[_key].set_qint(qint_file=parentPath/'data'/'level'/_path)
  else:
    raise Exception(f"The type {_type} is error.")
