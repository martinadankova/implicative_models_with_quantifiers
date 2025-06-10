import numpy as np
from . import FuzzyRelations as fr  # ✅ správný relativní import
import math as mt

def fourftable(Ax: np.ndarray, Bx: np.ndarray):
        nAx=fr.negSet(Ax)
        nBx=fr.negSet(Bx)
        outa=sum(np.multiply(Ax,Bx))
        outb=sum(np.multiply(Ax,nBx))
        outc=sum(np.multiply(nAx,Bx))
        outd=sum(np.multiply(nAx,nBx))
        outf=np.array([outa,outb,outc,outd])
        return outf

def QConfidence(a: np.real,b: np.real):
        outf=np.around(a/(a+b),decimals=2)
        return outf

def QimplL(a: np.real,b: np.real):
        outf=np.around(fr.implL(mt.pow(0.9,a+1),mt.pow(0.6,b+1)),decimals=2)
        return outf

def QimplP(a: np.real,b: np.real):
        outf=np.around(fr.implP(mt.pow(0.8,a+1),mt.pow(0.8,b+1)),decimals=2)
        return outf

def QImplRatioL(a: np.real,b: np.real):
        outf=np.around(fr.implL(b/(a+b),a/(b+a)),decimals=2)
        return outf

def QImplRatioP(a: np.real,b: np.real):
        outf=np.around(fr.implP(b/(a+b),a/(b+a)),decimals=2)
        return outf
