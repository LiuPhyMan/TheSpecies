import re
from math import pi, sqrt, exp
from typing import overload

import numpy as np
from myconst import (light_c, nm2m, Ry_eV, cm2m, h, epsilon_0, m_e, e,
                     eV2K, eV2J,
                     K2eV, bohr_radius as a0, fine_structure as fs_const,
                     nm2J_f, allSR, e2_eV)
from ._spc import _AbsSpecie
from mymath.basic import Kn
from mythermo.basic import rdcdM
# from mythermo import Composition as Comp
from scipy.interpolate import interp1d

reg = re.compile(r"""
^[^|]+\|\s*
(?P<wvlnm>\d+\.\d+)         # Wavelength in nm
[^|]*\|[^|]+\|\s+
(?P<Aki>[\d.e+-]+)           # Aki
\s*  \|  [^|]+  \|  \s+
(?P<Ei_eV>[\d.]+)           # Ei (eV)
\s*-\s*
(?P<Ek_eV>[\d.]+)           # Ek (eV)
\s*  \|  [^|]+  \|  [^|]+  \|  [^|]+  \|  \s*
(?P<Conf>\S+)              # Conf.
\s*  \|  \s*
(?P<Term>\S+)               # Term
\s*  \|  \s*
(?P<J>[\d/]+)                   # J
\s*  \|  [^|]+  \|  [^|]+  \|  [^|]+  \| 
\n
""", re.VERBOSE)


# class AtomLines(object):
#
#     def __init__(self, *, lineFile):
#         r"""
#         1. set_n_tot()
#         2. set_T()
#
#         Parameters
#         ----------
#         file_name
#         """


# =========================================================================== #
class _AbsDipoleTransition(object):
  __slots__ = ['spc', 'lvl', 'lin', 'sr']

  def __init__(self, *, spc: _AbsSpecie, lvl, lin):
    self.spc = spc
    self.lvl = lvl
    self.lin = lin
    self.sr = 1  # empirical factor.

  def set_sr(self, sr):
    self.sr = sr

  def lvl_alpha(self, iLvl):
    r"""polarizability of state i."""
    tmp = sum(self.prtb_R2(iLvl)/self.prtb_dE(iLvl))/eV2J
    return 2/3/pi*(h**2*epsilon_0/m_e/e)**2*tmp

  def norm_I(self, iLn, *, T_K):
    r""" h\nu * Aul * g*exp(-E/kT) / Q_tot / (4*pi)
    unit: W sr^-1"""
    lin_sr = self.lin.df.loc[iLn]
    norm_I = nm2J_f(lin_sr["wvlnm"])*lin_sr["Aul"]* \
             lin_sr["g_u"]*exp(-lin_sr["E_u"]*eV2K/T_K)/ \
             self.spc.qint(T_K)/allSR
    return norm_I

  def emiss(self, iLn, *, spc_n, T_K):
    r""" unit: W m^-3 sr^-1"""
    return spc_n*self.norm_I(iLn, T_K=T_K)

  def norm_I_tot(self, *, T_K):
    r""""""
    _df = self.lin.df
    return np.sum(nm2J_f(self.lin.df["wvlnm"])*self.lin.df["Aul"]* \
                  self.lin.df["g_u"]*np.exp(-self.lin.df["E_u"]*eV2K/T_K))/ \
      self.spc.qint(T_K)/allSR

  def tot_emiss(self, *, spc_n, T_K):
    return spc_n*self.norm_I_tot(T_K=T_K)

  # ------------------------------------------------------------------------- #
  def lin_R2(self, iLn, drct):
    r"""Square of the coordinate operator matrix element,
    in units of the Bohar radius a0^2."""
    rslt = 9.406593480824852e-07*self.lin.df.loc[iLn, "Aul"]/self.lin[
      iLn, "dE"]**3
    if drct=="up":
      return rslt
    elif drct=="down":
      return rslt*self.lin.df.loc[iLn, "g_u"]/self.lin.df.loc[iLn, "g_l"]
    else:
      raise Exception(drct)

  def prtb_R2(self, iLvl):
    lvl = self.lvl.df.loc[iLvl]
    return np.array([self.lin_R2(idx, 'up') for idx in lvl["lnIdx_uPrtb"]] + \
                    [self.lin_R2(idx, 'down') for idx in lvl["lnIdx_lPrtb"]])

  def prtb_dE(self, iLvl):
    lvl = self.lvl.df.loc[iLvl]
    return np.array(
      [-self.lin.df.loc[idx, "dE"] for idx in lvl["lnIdx_uPrtb"]] + \
      [self.lin.df.loc[idx, "dE"] for idx in lvl["lnIdx_lPrtb"]])

  def prtb_lvl_DK(self, iLvl, *, T_K):
    R2 = self.prtb_R2(iLvl)
    dE = np.abs(self.prtb_dE(iLvl))
    return np.dot(R2, self.gaunt_DK(dE*eV2K/3/T_K*np.sqrt(R2)))

  def shft_prtb_lvl_DK(self, iLvl, *, T_K):
    R2 = self.prtb_R2(iLvl)
    sign = -self.prtb_dE(iLvl)/np.abs(self.prtb_dE(iLvl))  # i -> i'
    dE = np.abs(self.prtb_dE(iLvl))
    return np.dot(R2, sign*self.gf_shft_DK(dE*eV2K/3/T_K*np.sqrt(R2)))

  def shft_prtb_lvl_Gr(self, iLvl, *, T_K):
    R2 = self.prtb_R2(iLvl)
    sign = self.prtb_dE(iLvl)/np.abs(self.prtb_dE(iLvl))  # i' -> i
    dE = np.abs(self.prtb_dE(iLvl))
    return np.dot(R2, sign*self.gf_shft_Gr(3*T_K/2/(dE*eV2K)))

  def prtb_lvl_Gr(self, iLvl, *, T_K):
    R2 = self.prtb_R2(iLvl)
    dE = np.abs(self.prtb_dE(iLvl))
    return np.dot(R2, [self.gaunt_Gr_ion(3*T_K/2/(_dE*eV2K)) for _dE in dE])

  def gaunt_DK(self, x):
    assert np.all(x >= 0), x
    return np.exp(-1.33*x)*np.log(1 + 2.27/x) + 0.487*x/(
        0.153 + x**(5/3)) + x/(7.93 + x**3)

  def gf_shft_Gr(self, x):
    assert np.all(x >= 0)
    return interp1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 60, 80, 100],
                    [.2, .25, .32, .41, .45, .51, .56, .60, .63, .66, .78, .82,
                     .84, .85,
                     .86, .87], kind="cubic", bounds_error=False,
                    fill_value=(.2, .87))(x)

  def gf_shft_DK(self, x):
    assert np.all(x >= 0)
    return 1.571*np.exp(-2.482*x) + 1.295*x/(0.415 + x**(5/3)) + 0.713*x/(
        8.139 + x**3)

  def gaunt_Gr_ion(self, x):
    if x <= 2:
      return 0.2
    elif x > 100:
      # raise Exception("")
      return 1.33
    else:
      return interp1d([2, 3, 5, 10, 30, 100],
                      [0.2, 0.24, 0.33, 0.56, 0.98, 1.33],
                      kind="cubic")(x)

  # ------------------------------------------------------------------------- #
  def ele_strk_shft_DK(self, iLn, *, comp, T_K):
    wvl = self.lin.df.loc[iLn, "wvlnm"]*nm2m
    prtb = self.shft_prtb_lvl_DK(self.lin.df.loc[iLn, "idx_lLvl"], T_K=T_K) - \
           self.shft_prtb_lvl_DK(self.lin.df.loc[iLn, "idx_uLvl"], T_K=T_K)
    shft = - self.sr*comp.ne*sqrt(
      8*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2*wvl**2*prtb
    return shft

  def ele_strk_shft_Gr(self, iLn, *, comp, T_K):
    wvl = self.lin.df.loc[iLn, "wvlnm"]*nm2m
    prtb = self.shft_prtb_lvl_Gr(self.lin.df.loc[iLn, "idx_uLvl"], T_K=T_K) - \
           self.shft_prtb_lvl_Gr(self.lin.df.loc[iLn, "idx_lLvl"], T_K=T_K)
    shft = comp.ne*sqrt(16*pi*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2*wvl**2*prtb
    return shft

  # ------------------------------------------------------------------------- #
  @overload
  def ele_strk_brdn_Gr(self, iLn, *, comp, T_K: float) -> float:
    pass

  @overload
  def ele_strk_brdn_Gr(self, iLn, *, ne: float, T_K: float) -> float:
    pass

  @overload
  def ele_strk_brdn_DK(self, iLn, *, comp, T_K: float) -> float:
    pass

  @overload
  def ele_strk_brdn_DK(self, iLn, *, ne: float, T_K: float) -> float:
    pass

  def ele_strk_brdn_Gr(self, iLn, **kwargs):
    assert kwargs.keys()=={"comp", "T_K"} or kwargs.keys()=={"ne", "T_K"}
    _ne = kwargs["comp"].ne if "comp" in kwargs else kwargs["ne"]
    T_K = kwargs["T_K"]
    prtb = self.prtb_lvl_Gr(self.lin[iLn, "idx_lLvl"], T_K=T_K) + \
           self.prtb_lvl_Gr(self.lin[iLn, "idx_uLvl"], T_K=T_K)
    HWHM = self.sr*_ne*sqrt(16*pi*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2* \
           (self.lin[iLn, "wvlnm"]*nm2m)**2*prtb
    return 2*HWHM

  def ele_strk_brdn_DK(self, iLn, **kwargs):
    assert kwargs.keys()=={"comp", "T_K"} or kwargs.keys()=={"ne", "T_K"}
    _ne = kwargs["comp"].ne if "comp" in kwargs else kwargs["ne"]
    T_K = kwargs["T_K"]
    prtb = self.prtb_lvl_DK(self.lin.df.loc[iLn, "idx_lLvl"], T_K=T_K) + \
           self.prtb_lvl_DK(self.lin.df.loc[iLn, "idx_uLvl"], T_K=T_K)
    HWHM = self.sr*_ne*sqrt(8*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2* \
           (self.lin[iLn, "wvlnm"]*nm2m)**2*prtb
    return 2*HWHM

  # ------------------------------------------------------------------------- #
  def ion_strk_brdn(self, iLn, *, ele_strk_brdn, comp):
    r"""stark_brdn = 2 * HWHM"""
    wvl = self.lin.df.loc[iLn, "wvlnm"]*nm2m
    tmp = abs(self.lvl_alpha(self.lin.df.loc[iLn, "idx_uLvl"]) - \
              self.lvl_alpha(self.lin.df.loc[iLn, "idx_lLvl"]))
    A = (comp.norm_field**2/2/h/light_c*wvl**2/(ele_strk_brdn/2)*tmp)**(3/4)
    R = comp.ion_distn/comp.totDebL()
    # assert A <= 0.5, A,   # TODO
    # assert 0.05 <= A , A
    # assert R <= 0.8, R # TODO.
    assert self.spc.Zc >= 0
    xi = 1.25 if self.spc.Zc > 0 else 0.75
    return ele_strk_brdn*1.75*A*(1 - xi*R)

  # ------------------------------------------------------------------------- #
  def vdw_brdn(self, iLn, *, comp, T_K: float) -> float:
    if not self.spc.isNeu():
      return 0.0
    wvlnm = self.lin.df.loc[iLn, "wvlnm"]
    # n2_u = Ry_eV/(self.spc.ionE - self.lvl[self.lin[iLn, "idx_uLvl"], "E_eV"])
    # n2_l = Ry_eV/(self.spc.ionE - self.lvl[self.lin[iLn, "idx_lLvl"], "E_eV"])
    n2_u = Ry_eV/self.lvl[self.lin[iLn, 'idx_uLvl'], 'Ebind0']
    n2_l = Ry_eV/self.lvl[self.lin[iLn, 'idx_lLvl'], 'Ebind0']
    assert (n2_u > 0) and (n2_l > 0)
    l_u = self.lvl[self.lin[iLn, "idx_uLvl"], "l"]
    l_l = self.lvl[self.lin[iLn, "idx_lLvl"], "l"]
    R2_u = n2_u/2*(5*n2_u + 1 - 3*l_u*(l_u + 1))
    R2_l = n2_l/2*(5*n2_l + 1 - 3*l_l*(l_l + 1))
    assert R2_u > 0, R2_u
    assert R2_l > 0, R2_l
    R2 = abs(R2_u - R2_l)
    # ---
    tmp = 0
    for i in range(comp.n_spcs):
      if comp.Zc[i]==0:
        tmp = tmp + comp.spcs[i].polar**0.4* \
              rdcdM(self.spc.relM, comp.spcs[i].relM)**(-0.3)* \
              comp.nj[i]*comp.spcs[i].gnd_frac(T_K)
    _const = 9.573586447390544e-42
    return _const*wvlnm**2*R2**0.4*T_K**0.3*tmp*cm2m

  def res_brdn(self, iLn, *, comp, T_K):
    gnd_nj = comp.nj_dict[self.spc.spc_str]*self.spc.gnd_frac(T_K)
    return self.lin.df.loc[iLn, "res_const"]*gnd_nj

  def dpp_brdn(self, iLn, *, T_K):
    return 7.162326482387328e-07*self.lin.df.loc[iLn, "wvlnm"]*nm2m* \
      sqrt(T_K/self.spc.relM)

  def gauss_brdn(self, iLn, *, T_K):
    return self.dpp_brdn(iLn, T_K=T_K)

  def loren_brdn(self, iLn, *, comp, T_K, strk_method="DK"):
    # assert strk_method == "DK" #TODO
    if strk_method=="DK":
      ele_strk = self.ele_strk_brdn_DK(iLn, comp=comp, T_K=T_K)
      ion_strk = self.ion_strk_brdn(iLn, ele_strk_brdn=ele_strk, comp=comp)
      return self.res_brdn(iLn, comp=comp, T_K=T_K) + \
        self.vdw_brdn(iLn, comp=comp, T_K=T_K) + ele_strk + ion_strk
    else:
      return self.ele_strk_brdn_Gr(iLn, comp=comp, T_K=T_K)


class DipoleTransition(_AbsDipoleTransition):

  def __init__(self, *, spc, lvl, lin):
    super().__init__(spc=spc, lvl=lvl, lin=lin)

  def norm_I(self, iLn, *, T_K, ipd=0):
    # TODO
    return super().norm_I(iLn, T_K=T_K)

  def isLvlFree(self, iLn, *, ipd):
    r"""Ionization potential is not same for every level."""
    if self.lvl[iLn, 'Ebind0'] < ipd:
      return True
    else:
      return False

  def probCollct(self, iLvl, *, Zi, ni, ipd=0):
    r""""""
    Z0 = self.spc.Zc + 1  # charge of the parent ion.
    Ebind = self.lvl[iLvl, 'Ebind0'] - ipd
    assert Ebind >= 0
    rCrit = sqrt(4*Z0*Zi*e2_eV**2/Kn(self.lvl[iLvl, 'n'])/Ebind**2)
    empirical_factor = 2
    return exp(-4*pi/3*rCrit**3*ni*empirical_factor)
