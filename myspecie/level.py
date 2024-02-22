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
# class PhotoIonizationCS(object):
#
#     def __init__(self, *, spc, lvl_df):
#         self.spc = spc
#         self.lvl_df = lvl_df
#
#     def norm_cs(self, wvlnm, *, T_K):
#         _cs = np.zeros_like(wvlnm)
#         for iLvl in self.lvl_df.index:
#             pass
# from math import pi, sqrt, exp
# import numpy as np
# from myconst import fine_structure as fs_const, bohr_radius as a0, Ry_eV,
# nm2Hz_f, Hz2nm_f, eV2K
#
#
# T_K = 10e3
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
