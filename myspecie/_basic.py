# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 03.Jan.2023
"""
from math import exp

import numpy as np
from myconst import eV2K


class Lines(object):
  __slots__ = ['df']

  def __init__(self, df):
    self.df = df

  def __getitem__(self, _paras):
    r""" _paras: (index, column)"""
    return self.df.loc[_paras[0], _paras[1]]

  @property
  def index(self):
    return self.df.index


class Levels(object):
  __slots__ = ['df']

  def __init__(self, df):
    self.df = df

  def __getitem__(self, _paras):
    r""" _paras: (index, column)
    access:   lvl[iLvl, column] """
    return self.df.loc[_paras[0], _paras[1]]

  def Zi(self, iLvl, *, T_K):
    _ser = self.df.loc[iLvl]
    return _ser['g']*exp(-_ser['E_eV']*eV2K/T_K)

  def Z(self, *, T_K):
    return np.dot(self.df['g'], np.exp(-self.df['E_eV']*eV2K/T_K))

  def normZi(self, iLvl, *, T_K):
    return self.Zi(iLvl, T_K=T_K)/self.Z(T_K=T_K)


def atomStrs(atom, *, maxZc):
  r"""Generate a list of atom str.
  e.g.
  ---
  [Ar, Ar_1p, Ar_2p, ...]
  """
  assert maxZc >= 1
  return [atom] + [f"{atom}_{Zc:.0f}p" for Zc in range(1, maxZc + 1)]
