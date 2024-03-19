# Author:   Liujinbao
# Email:    liu.jinbao@outlook.com
# Time:     2024.01.12

# from math import pi, sqrt
# import numpy as np
# from myconst import fine_structure as fs_const, bohr_radius as a0, Ry_eV
#
# _const = 64*pi*fs_const*a0**2/3/sqrt(3)
#
#
from math import pi, sqrt

from myconst import fine_structure as fs_const, bohr_radius as a0, Ry_eV as \
  Ry, nm2eV_f, eV2nm_f
import numpy as np

from ._basic import Levels


class PhotoIonizationCS(object):
  def __init__(self, *, lvl: Levels):
    self.lvl = lvl

  def cs(self, iLvl, *, spl_wvlnm, ipd):
    Qn = self.lvl[iLvl, 'Qn']
    Ebind = self.lvl[iLvl, 'Ebind0'] - ipd
    if Ebind < 0:
      return np.zeros_like(spl_wvlnm)
    else:
      _const = 64*pi*fs_const*a0**2/3/sqrt(3)
      tmp = _const*Ebind**2.5*sqrt(Ry)/nm2eV_f(spl_wvlnm)**3
      tmp[spl_wvlnm > eV2nm_f(Ebind)] = 0
      return tmp/Qn

  def norm_cs(self):
    pass

# _const = 64*pi*fs_const*a0**2/3/sqrt(3)
#
# nuHz_seq = np.linspace(0.1, 10, num=10000)*1e15
# wvlnm_seq = Hz2nm_f(nuHz_seq)
# # wvlnm_seq = np.linspace(1, 20000, num=10000)
# norm_cs_seq = np.zeros_like(wvlnm_seq)
# for iLvl in lvl_df.index:
#     if lvl_df.loc[iLvl, 'Eth_nm'] < wvlnm_seq.min():
#         continue
#     _cs = np.zeros_like(wvlnm_seq)
#     _cs[wvlnm_seq<lvl_df.loc[iLvl, "Eth_nm"]] = _const/lvl_df.loc[iLvl,
#     "Qn"]*sqrt(Ry_eV/lvl_df.loc[iLvl, 'Eth_eV'])*\
#         (wvlnm_seq[wvlnm_seq<lvl_df.loc[iLvl, "Eth_nm"]]/lvl_df.loc[iLvl,
#         'Eth_nm'])**3
#     norm_cs_seq = norm_cs_seq + _cs*lvl_df.loc[iLvl, 'g']*exp(-lvl_df.loc[
#     iLvl, "E"]*eV2K/T_K)
