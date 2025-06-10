import numpy as np

def negSet(x: np.ndarray):
        outf=1-x
        return outf

def implM(x: np.ndarray, y: np.ndarray):
        outf=np.where(x<=y,1,y)
        return outf

def conjM(x: np.ndarray, y: np.ndarray):
        outf=np.minimum(x,y)
        return outf

def implP(x: np.ndarray, y: np.ndarray):
        outf=np.where(x<=y,1,y/x)
        return outf

def conjP(x: np.ndarray, y: np.ndarray):
        outf=x*y
        return outf

def implL(x: np.ndarray, y: np.ndarray):
        outf=1-x+y
        outf=np.minimum(1,outf)
        return outf

def conjL(x: np.ndarray, y: np.ndarray):
        outf=x+y-1
        outf=np.maximum(0,outf)
        return outf

def frel(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Fuzzy relation used for describing neighbourhood R(x,y)=0v(1-|x-y|)
        xx, yy = np.meshgrid(x, y)
        outfrel =np.absolute(np.subtract(xx, yy))
        outfrel = 1-outfrel
        outfrel=np.maximum(0,outfrel)
        return outfrel

def CartL(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Lukasiewicz cartesian product
        xx, yy = np.meshgrid(x, y)
        outfrel =np.add(xx, yy)
        outfrel = outfrel-1
        outfrel=np.maximum(0,outfrel)
        outfrel=np.minimum(1,outfrel)
        return outfrel

def CartM(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Lukasiewicz cartesian product
        xx, yy = np.meshgrid(x, y)
        outfrel =np.minimum(xx, yy)
        return outfrel

def CartP(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Lukasiewicz cartesian product
        xx, yy = np.meshgrid(x, y)
        outfrel =xx*yy
        return outfrel

def CartImplP(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Lukasiewicz implicational product
        xx, yy = np.meshgrid(x, y)
        outfrel = implP(xx,yy)
        return outfrel

def CartImplL(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Lukasiewicz implicational product
        xx, yy = np.meshgrid(x, y)
        outfrel = 1-xx
        outfrel =np.add(outfrel, yy)
        outfrel=np.maximum(0,outfrel)
        outfrel=np.minimum(1,outfrel)
        return outfrel

def frelATL(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Fuzzy relation at least R(x,y)=0v(1-(x-y))
        xx, yy = np.meshgrid(x, y)
        outfrel =np.subtract(xx, yy)
        outfrel = 1-outfrel
        outfrel=np.maximum(0,outfrel)
        outfrel=np.minimum(1,outfrel)
        return outfrel

def frelATM(x: np.ndarray, y: np.ndarray): #-> np.ndarray:
#    Fuzzy relation at most R(x,y)=0v(1-(y-x))
        xx, yy = np.meshgrid(x, y)
        outfrel =np.subtract(yy,xx)
        outfrel = 1-outfrel
        outfrel=np.maximum(0,outfrel)
        outfrel=np.minimum(1,outfrel)
        return outfrel

def ksim(x: np.ndarray, y: np.ndarray, k: np.int0): #-> np.ndarray:
#    Fuzzy similarity used for describing neighbourhood S_k(x,y)=0v(1-k|x-y|)
        xx, yy = np.meshgrid(x, y)
        outksim =np.absolute(np.subtract(xx, yy))
        outksim = 1-k*outksim
        outksim=np.maximum(0,outksim)
        return outksim

def pointksim(x: np.ndarray, k: np.real,c: np.real): #-> np.ndarray:
#    Fuzzy similarity used for describing neighbourhood of c, i.e., S_k(x,c)=[0v(1-k|x-c|)]
        outfx =np.subtract(x,c)
        outfx =np.absolute(outfx)
        outfx = 1-k*outfx
        outfx=np.maximum(0,outfx)
        return outfx

def finterval(x: np.ndarray,c: np.real,d: np.real, k: np.real): #-> np.ndarray:
#    Fuzzy interval, i.e., c\leq_k x&x\leq_k d 
        outfx =atleast(x,c,k)*atmost(x,d,k)
        return outfx

def fintervalM(x: np.ndarray,c: np.real,d: np.real, k: np.real): #-> np.ndarray:
#    Fuzzy interval, i.e., c\leq_k x min x\leq_k d 
        outfx =atleast(x,c,k)*atmost(x,d,k)
        return outfx

def atmost(x: np.ndarray,c: np.real, k: np.real): #-> np.ndarray:
#    Fuzzy at most relation, i.e., c\leq_k x = min[0v(1-k(x-c)),1]
        outfx =np.subtract(x,c)
        outfx = 1-k*outfx
        outfx=np.maximum(0,outfx)
        outfx=np.minimum(1,outfx)
        return outfx

def atleast(x: np.ndarray,c: np.real, k: np.real): #-> np.ndarray:
#    Fuzzy at least relation of c, i.e., c\leq_k x = min[0v(1-k(c-x)),1]
        outfx =np.subtract(c,x)
        outfx = 1-k*outfx
        outfx=np.maximum(0,outfx)
        outfx=np.minimum(1,outfx)
        return outfx
