import re
from math import pi

from myconst import c as light_c
from pandas import DataFrame

reg = re.compile(r"""
^[^|]+\|\s*
(?P<wvlth>\d+\.\d+)    # Wavelength
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
        columns = ["wvlth", "Aki", "Ei_eV", "Ek_eV", "Conf", "Term", "J"]
        # -- read file to data -- #
        for line in lines:
            temp = reg.fullmatch(line)
            if temp:
                data.append([temp.group(_) for _ in columns])
        self.df = DataFrame(data, columns=columns)
        # -- str to number -- #
        self.df["wvlth"] = self.df["wvlth"].map(lambda x: float(x)*1e-9)
        # self.wvlth = df["wvlth"].map(lambda x: float(x)*1e-9)
        self.df["Aki"] = self.df["Aki"].map(lambda x: float(x))
        self.df["Ei_eV"] = self.df["Ei_eV"].map(lambda x: float(x))
        self.df["Ek_eV"] = self.df["Ek_eV"].map(lambda x: float(x))
        self.df["J"] = self.df["J"].map(lambda x: float(eval(x)))
        self.df["g"] = self.df["J"]*2 + 1
        self.df["nu"] = light_c/self.df["wvlth"]
        self.df["omega"] = 2*pi*self.df["nu"]
        self.df["wAg"] = self.df["omega"]*self.df["Aki"]*self.df["g"]
        #
        self.n_lines = self.df["wvlth"].size

    def df_filter_by_wv_range(self, *, wvlth_range):
        df = self.df[(self.df["wvlth"] >= wvlth_range[0])* (self.df["wvlth"] <= wvlth_range[1])]
        # assert self.df.index.size > 0, "Empty line"
        return df
        # self.n_lines = self.df.index.size

    def __repr__(self):
        return f"{self.n_lines} lines"
