import re
from math import pi, sqrt, exp

import numpy as np
import pandas as pd
from myconst import (c as light_c, nm2m, Ry_eV, cm2m, h, epsilon_0, m_e, e, eV2K, eV2J,
                     K2eV, bohr_radius as a0, fine_structure as fs_const, nm2Hz_f, allSR, Hz2J)
# from mythermo.Comp import Composition
from mythermo.basic import rdcdM
from pandas import DataFrame
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


class AtomLines(object):

    def __init__(self, *, lineFile):
        r"""
        1. set_n_tot()
        2. set_T()

        Parameters
        ----------
        file_name
        """
        self._file_name = lineFile
        with open(lineFile) as f:
            lines = f.readlines()
        data = []
        columns = ["wvlnm", "Aki", "Ei_eV", "Ek_eV", "Conf", "Term", "J"]
        # -- read file to data -- #
        for line in lines:
            temp = reg.fullmatch(line)
            if temp:
                data.append([temp.group(_) for _ in columns])
        self.df = DataFrame(data, columns=columns)
        # -- str to number -- #
        self.df["wvlnm"] = self.df["wvlnm"].map(lambda x: float(x))
        self.df["Aki"] = self.df["Aki"].map(lambda x: float(x))
        self.df["Ei_eV"] = self.df["Ei_eV"].map(lambda x: float(x))
        self.df["Ek_eV"] = self.df["Ek_eV"].map(lambda x: float(x))
        self.df["J"] = self.df["J"].map(lambda x: float(eval(x)))
        self.df["g"] = self.df["J"]*2 + 1
        self.df["nu"] = light_c/(self.df["wvlnm"]*nm2m)
        self.df["omega"] = 2*pi*self.df["nu"]
        self.df["wAg"] = self.df["omega"]*self.df["Aki"]*self.df["g"]
        #
        self.n_lines = self.df["wvlnm"].size

    def df_filter_by_wv_range(self, *, wvlnm_rng):
        df = self.df[(self.df["wvlnm"] >= wvlnm_rng[0])*(self.df["wvlnm"] <= wvlnm_rng[1])]
        return df

    def __repr__(self):
        return f"{self.n_lines} lines"


# ----------------------------------------------------------------------------------------------- #

class AtomLinesKurucz(object):

    def __init__(self, *, df_file):
        self.df = pd.read_pickle(df_file)

    def set_wvlnm_rng(self, wvl_nm_l, wvl_nm_u):
        self.wvlnm_rng = (wvl_nm_l, wvl_nm_u)

    def set_sr(self, sr: float):
        self.sr = sr

    # def wvl_filter(self, *, wvlnm_rng):
    # df = self.df[(self.df["wvlnm"] >= wvlnm_rng[0])* (self.df["wvlnm"] <= wvlnm_rng[1])]
    # return df
    def set_stark_brdn_DK(self, *, ne: float, T_K: float):
        _const = 4.4198201551008605e-39
        self.df["stark_brdn"] = 0
        for i in self.df.index:
            if (self.df.loc[i, "wvlnm"] < self.wvlnm_rng[0]) or (self.df.loc[i, "wvlnm"] >
                                                                 self.wvlnm_rng[1]):
                continue
            else:
                prtb = self.df.loc[i, "prtb_u"] + self.df.loc[i, "prtb_l"]
                tmp = _const*self.sr*ne*self.df.loc[i, "wvlnm"]/sqrt(T_K)
                # tmp1 =


# =============================================================================================== #
class DipoleTransition(object):

    def __init__(self, *, spc, lvl_df, lin_df):
        self.spc = spc
        self.lvl_df = lvl_df
        self.lin_df = lin_df
        self.sr = 1

    def set_sr(self, sr):
        self.sr = sr

    def lvl_alpha(self, iLvl):
        r"""polarizability of state i."""
        tmp = sum(self.prtb_R2(iLvl)/self.prtb_dE(iLvl))/eV2J
        return 2/3/pi*(h**2*epsilon_0/m_e/e)**2*tmp

    def norm_I(self, iLn, *, T_K):
        r""" h\nu * Aul * g*exp(-E/kT) / Q_tot / (4*pi)"""
        norm_I = nm2Hz_f(self.lin_df.loc[iLn, "wvlnm"])*Hz2J*self.lin_df.loc[iLn, "Aul"]* \
                 self.lin_df.loc[iLn, "g_u"]*exp(-self.lin_df.loc[iLn, "E_u"]*eV2K/T_K)/ \
                 self.spc.qint(T_K)/allSR
        return norm_I

    # ------------------------------------------------------------------------------------------- #
    def lin_R2(self, iLn, drct):
        r"""Square of the coordinate operator matrix element,
        in units of the Bohar radius a0^2."""
        rslt = 9.406593480824852e-07*self.lin_df.loc[iLn, "Aul"]/self.lin_df.loc[iLn, "dE"]**3
        if drct == "up":
            return rslt
        elif drct == "down":
            return rslt*self.lin_df.loc[iLn, "g_u"]/self.lin_df.loc[iLn, "g_l"]
        else:
            raise Exception(drct)

    def prtb_R2(self, iLvl):
        lvl = self.lvl_df.loc[iLvl]
        return np.array([self.lin_R2(idx, 'up') for idx in lvl["lnIdx_uPrtb"]] + \
                        [self.lin_R2(idx, 'down') for idx in lvl["lnIdx_lPrtb"]])

    def prtb_dE(self, iLvl):
        lvl = self.lvl_df.loc[iLvl]
        return np.array([-self.lin_df.loc[idx, "dE"] for idx in lvl["lnIdx_uPrtb"]] + \
                        [self.lin_df.loc[idx, "dE"] for idx in lvl["lnIdx_lPrtb"]])

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
        return np.exp(-1.33*x)*np.log(1 + 2.27/x) + 0.487*x/(0.153 + x**(5/3)) + x/(7.93 + x**3)

    def gf_shft_Gr(self, x):
        assert np.all(x >= 0)
        return interp1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 60, 80, 100],
                        [.2, .25, .32, .41, .45, .51, .56, .60, .63, .66, .78, .82, .84, .85,
                         .86, .87], kind="cubic", bounds_error=False, fill_value=(.2, .87))(x)

    def gf_shft_DK(self, x):
        assert np.all(x >= 0)
        return 1.571*np.exp(-2.482*x) + 1.295*x/(0.415 + x**(5/3)) + 0.713*x/(8.139 + x**3)

    def gaunt_Gr_ion(self, x):
        if x <= 2:
            return 0.2
        elif x > 100:
            # raise Exception("")
            return 1.33
        else:
            return interp1d([2, 3, 5, 10, 30, 100], [0.2, 0.24, 0.33, 0.56, 0.98, 1.33],
                            kind="cubic")(x)

    # ------------------------------------------------------------------------------------------- #
    def ele_strk_shft_DK(self, iLn, *, comp, T_K):
        wvl = self.lin_df.loc[iLn, "wvlnm"]*nm2m
        prtb = self.shft_prtb_lvl_DK(self.lin_df.loc[iLn, "idx_lLvl"], T_K=T_K) - \
               self.shft_prtb_lvl_DK(self.lin_df.loc[iLn, "idx_uLvl"], T_K=T_K)
        shft = - self.sr*comp.ne*sqrt(8*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2*wvl**2*prtb
        return shft

    def ele_strk_brdn_DK(self, iLn, *, comp, T_K):
        prtb = self.prtb_lvl_DK(self.lin_df.loc[iLn, "idx_lLvl"], T_K=T_K) + \
               self.prtb_lvl_DK(self.lin_df.loc[iLn, "idx_uLvl"], T_K=T_K)
        HWHM = self.sr*comp.ne*sqrt(8*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2* \
               (self.lin_df.loc[iLn, "wvlnm"]*nm2m)**2*prtb
        return 2*HWHM

    def ele_strk_shft_Gr(self, iLn, *, comp, T_K):
        wvl = self.lin_df.loc[iLn, "wvlnm"]*nm2m
        prtb = self.shft_prtb_lvl_Gr(self.lin_df.loc[iLn, "idx_uLvl"], T_K=T_K) - \
               self.shft_prtb_lvl_Gr(self.lin_df.loc[iLn, "idx_lLvl"], T_K=T_K)
        shft = comp.ne*sqrt(16*pi*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2*wvl**2*prtb
        return shft

    def ele_strk_brdn_Gr(self, iLn, *, comp, T_K):
        prtb = self.prtb_lvl_Gr(self.lin_df.loc[iLn, "idx_lLvl"], T_K=T_K) + \
               self.prtb_lvl_Gr(self.lin_df.loc[iLn, "idx_uLvl"], T_K=T_K)
        HWHM = self.sr*comp.ne*sqrt(16*pi*Ry_eV/27/(T_K*K2eV))*fs_const*a0**2* \
               (self.lin_df.loc[iLn, "wvlnm"]*nm2m)**2*prtb
        return 2*HWHM

    def ion_strk_brdn(self, iLn, *, ele_strk_brdn, comp):
        r"""stark_brdn = 2 * HWHM"""
        wvl = self.lin_df.loc[iLn, "wvlnm"]*nm2m
        tmp = abs(self.lvl_alpha(self.lin_df.loc[iLn, "idx_uLvl"]) - \
                  self.lvl_alpha(self.lin_df.loc[iLn, "idx_lLvl"]))
        A = (comp.norm_field**2/2/h/light_c*wvl**2/(ele_strk_brdn/2)*tmp)**(3/4)
        R = comp.ion_distn/comp.DebL()
        # assert A <= 0.5, A,   # TODO
        # assert 0.05 <= A , A
        # assert R <= 0.8, R # TODO.
        assert self.spc.Zc >= 0
        xi = 1.25 if self.spc.Zc > 0 else 0.75
        return ele_strk_brdn*1.75*A*(1 - xi*R)

    # ------------------------------------------------------------------------------------------- #
    def vdw_brdn(self, iLn, *, comp, T_K: float) -> float:
        wvlnm = self.lin_df.loc[iLn, "wvlnm"]
        n2_u = Ry_eV/(self.spc.ionE - self.lvl_df.loc[self.lin_df.loc[iLn, "idx_uLvl"], "E_eV"])
        n2_l = Ry_eV/(self.spc.ionE - self.lvl_df.loc[self.lin_df.loc[iLn, "idx_lLvl"], "E_eV"])
        l_u = self.lvl_df.loc[self.lin_df.loc[iLn, "idx_uLvl"], "l"]
        l_l = self.lvl_df.loc[self.lin_df.loc[iLn, "idx_lLvl"], "l"]
        R2_u = n2_u/2*(5*n2_u + 1 - 3*l_u*(l_u + 1))
        R2_l = n2_l/2*(5*n2_l + 1 - 3*l_l*(l_l + 1))
        R2 = abs(R2_u - R2_l)
        # ---
        tmp = 0
        for i in range(comp.n_spcs):
            if comp.Zc[i] == 0:
                tmp = tmp + comp.spcs[i].polar**0.4* \
                      rdcdM(self.spc.relM, comp.spcs[i].relM)**(-0.3)* \
                      comp.nj[i]*comp.spcs[i].gnd_frac(T_K)
        _const = 9.573586447390544e-42
        return _const*wvlnm**2*R2**0.4*T_K**0.3*tmp*cm2m

    def res_brdn(self, iLn, *, comp, T_K):
        gnd_nj = comp.nj_dict[self.spc.spc_str]*self.spc.gnd_frac(T_K)
        return self.lin_df.loc[iLn, "res_const"]*gnd_nj

    def dpp_brdn(self, iLn, *, T_K):
        return 7.162326482387328e-07*self.lin_df.loc[iLn, "wvlnm"]*nm2m* \
            sqrt(T_K/self.spc.relM)

    def gauss_brdn(self, iLn, *, T_K):
        return self.dpp_brdn(iLn, T_K=T_K)

    def loren_brdn(self, iLn, *, comp, T_K, strk_method="DK"):
        # assert strk_method == "DK"
        if strk_method == "DK":
            ele_strk = self.ele_strk_brdn_DK(iLn, comp=comp, T_K=T_K)
            ion_strk = self.ion_strk_brdn(iLn, ele_strk_brdn=ele_strk, comp=comp)
            return self.res_brdn(iLn, comp=comp, T_K=T_K) + \
                self.vdw_brdn(iLn, comp=comp, T_K=T_K) + ele_strk + ion_strk
        else:
            return self.ele_strk_brdn_Gr(iLn, comp=comp, T_K=T_K)
