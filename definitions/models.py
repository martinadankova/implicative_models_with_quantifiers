import numpy as np
from . import FuzzyRelations as fr  # relativní import
import pandas as pd
import plotly.graph_objects as go
from . import quantifiers as qt
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pickle

def defuzzify_cog(x, mu):
    return np.sum(x * mu) / np.sum(mu)

def defuzzify_mom(x, mu):
    x = np.asarray(x)
    max_mu = np.max(mu)
    return np.mean(x[np.isclose(mu, max_mu)])

def defuzzify_maxom(x, mu):
    x = np.asarray(x)
    max_mu = np.max(mu)
    return np.max(x[np.isclose(mu, max_mu)])

def defuzzify_minom(x, mu):
    x = np.asarray(x)
    max_mu = np.max(mu)
    return np.min(x[np.isclose(mu, max_mu)])

def plot_and_save_QModel(model_output, filename_base="QModel"):
    # Rozbal parametry
    x = model_output["x"]
    y = model_output["y"]
    ModelQuantifiedRules = model_output["ModelQuantifiedRules"]
    nodesx = model_output["nodesx"]
    nodesyL = model_output["nodesyL"]
    nodesyR = model_output["nodesyR"]
    quantifier = model_output["quantifier"]
    
    # === 1. Vykreslení fuzzy modelu ===
    plt.figure(figsize=(8, 6))
    extent = [min(x), max(x), min(y), max(y)]
    plt.imshow(ModelQuantifiedRules, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Membership Degree')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Quantified Fuzzy Model")
    plt.tight_layout()
    
    # Uložení grafu
    plot_path = f"{filename_base}_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # === 2. Uložení modelu jako pickle ===
    pickle_path = f"{filename_base}_model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(model_output, f)

    return {"plot_file": plot_path, "pickle_file": pickle_path}

def load_and_plot_QModel(pickle_path):
    # Načti model ze souboru
    with open(pickle_path, "rb") as f:
        model_output = pickle.load(f)

    # Rozbal parametry
    x = model_output["x"]
    y = model_output["y"]
    ModelQuantifiedRules = model_output["ModelQuantifiedRules"]

    # Vykreslení
    plt.figure(figsize=(8, 6))
    extent = [min(x), max(x), min(y), max(y)]
    plt.imshow(ModelQuantifiedRules, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Membership Degree')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Loaded Quantified Fuzzy Model")
    plt.tight_layout()
    plt.show()

def QRules_model(datax: np.ndarray, datavalx: np.ndarray, dx: int, decl: int):
    maxx = np.max(datax)
    maxfx = np.max(datavalx)
    minx = np.min(datax)
    minfx = np.min(datavalx)

    nodesx = [minx + k * ((maxx - minx) / dx) for k in range(dx)]
    nodesx.append(maxx + 1)

    disx = 100
    x = [minx + k * ((maxx - minx) / disx) for k in range(disx)]

    disy = 100
    y = [minfx + k * ((maxfx - minfx) / disy) for k in range(disy)]

    # create fuzzy intervals and quantifiers
    nodesyL = []
    nodesyR = []
    quantifier = []

    for i in range(len(nodesx) - 1):
        start_interval = nodesx[i]
        end_interval = nodesx[i + 1]
        weight = fr.fintervalM(datax, start_interval, end_interval, decl / maxx)
        sum_weight = np.sum(weight)

        weighted_mean = np.sum(weight * datavalx) / sum_weight
        weighted_variance = np.sum(weight * np.abs(datavalx - weighted_mean)) / sum_weight

        left = weighted_mean - weighted_variance
        right = weighted_mean + weighted_variance
        nodesyL.append(left)
        nodesyR.append(right)

        Ai = fr.fintervalM(datax, start_interval, end_interval, decl / maxx)
        Bi = fr.fintervalM(datavalx, left, right, decl / maxfx)
        table_AiBi = qt.fourftable(Ai, Bi)
        q1 = qt.QConfidence(table_AiBi[0], table_AiBi[1])
        quantifier.append(q1)

    # construct the final quantified model
    A1x = fr.fintervalM(x, nodesx[0], nodesx[1], decl / maxx)
    B1y = fr.fintervalM(y, nodesyL[0], nodesyR[0], decl / maxfx)
    ModelRules = fr.CartImplL(A1x, B1y)
    ModelQuantifiedRules = fr.implL(quantifier[0], ModelRules)

    for i in range(1, len(nodesyL)):
        Aix = fr.fintervalM(x, nodesx[i], nodesx[i+1], decl / maxx)
        Biy = fr.fintervalM(y, nodesyL[i], nodesyR[i], decl / maxfx)
        ModelRules = fr.CartImplL(Aix, Biy)
        ModelQuantifiedRules = np.minimum(ModelQuantifiedRules, fr.implL(quantifier[i], ModelRules))

    return {
        "ModelQuantifiedRules": ModelQuantifiedRules,
        "nodesx": nodesx,
        "nodesyL": nodesyL,
        "nodesyR": nodesyR,
        "x": x,
        "y": y,
        "quantifier": quantifier,
        "maxx": maxx,
        "minx": minx,
        "maxfx": maxfx,
        "minfx": minfx,
        "decl": decl,
        "dx": dx
    }

def QRules(datax:np.ndarray, datavalx: np.ndarray,dx:np.integer,decl:np.integer):
    maxx=max(datax)
    maxfx=max(datavalx)
    minx=min(datax)
    minfx=min(datavalx)
    nodesx = [minx + k * ((maxx - minx) / dx) for k in range(0,dx)]
    nodesx.append(maxx+1)
    # set discretization of X and Y for final plots
    disx=100
    x=[minx+k*((maxx-minx)/disx) for k in range(0,disx)]
    disy=100
    y=[minfx+k*((maxfx-minfx)/disy) for k in range(0,disy)]

    # create fuzzy intervals on X for discretization of X
    # plot fuzzy sets on X
    fig1 = plt.figure()
    for i in range(0, len(nodesx)-1):
        Aix=fr.fintervalM(x,nodesx[i],nodesx[i+1],decl/maxx)
        plt.plot(x,Aix)

    plt.show()

    nodesyL = []  # Initialize the list to store weighted means-variance
    nodesyR = []  # Initialize the list to store weighted means+variance

    # Calculate weighted means for each interval
    for i in range(len(nodesx) - 1):
        start_interval = nodesx[i]
        end_interval = nodesx[i + 1]
        # Calculate weights using fr.fintervalM
        weight = fr.fintervalM(datax, start_interval, end_interval, decl / maxx)
        sum_weight=sum(weight)
        # Calculate the weighted mean for the interval
        weighted_mean = sum(weight * datavalx) / sum_weight
        weighted_variance=sum(weight * abs(datavalx-weighted_mean)) / sum_weight
        nodesyL.append(weighted_mean-weighted_variance)
        nodesyR.append(weighted_mean+weighted_variance)

    print(nodesyL)
    print(nodesyR)

    for i in range(len(nodesyL)):
        Biy=fr.fintervalM(y,nodesyL[i],nodesyR[i],decl/maxfx)
        plt.plot(y,Biy)
    plt.show()

    #calculate membershipvalues of data to fuzzy sets on X and Y
    quantifier=[]
    for i in range(len(nodesyL)):
        Ai=fr.fintervalM(datax,nodesx[i],nodesx[i+1],decl/maxx)
        Bi=fr.fintervalM(datavalx,nodesyL[i],nodesyR[i],decl/maxfx)
        table_AiBi=qt.fourftable(Ai,Bi)
        q1=qt.QConfidence(table_AiBi[0],table_AiBi[1])
        quantifier.append(q1)
    print(quantifier)

    A1x=fr.fintervalM(x,nodesx[0],nodesx[1],decl/maxx)
    B1y=fr.fintervalM(y,nodesyL[0],nodesyR[0],decl/maxfx)
    ModelRules=fr.CartImplL(A1x,B1y)
    ModelQuantifiedRules=fr.implL(quantifier[0],ModelRules)
    p=ShowModelwithData(x,y,ModelQuantifiedRules,datax,datavalx,datax*0
                             ,'Quantifier Based Implicative Rule 1')
    plt.show
    for i in range(1,len(nodesyL)):
        print(i)
        Aix=fr.fintervalM(x,nodesx[i],nodesx[i+1],decl/maxx)
        Biy=fr.fintervalM(y,nodesyL[i],nodesyR[i],decl/maxfx)
        ModelRules=fr.CartImplL(Aix,Biy)
        ModelQuantifiedRules=np.minimum(ModelQuantifiedRules,fr.implL(quantifier[i],ModelRules))
        plt.show
        p=ShowModelwithData(x,y,ModelQuantifiedRules,datax,datavalx,datax*0
                             ,'Quantifier Based Implicative Rule - i:'+str(i+1))
        plt.show

    return ModelQuantifiedRules

def MamdATLxATMy(data_x: np.ndarray, data_fx: np.ndarray,disx:np.integer,disy:np.integer):
    # Atleast Atmost mamdani model
    maxx=max(data_x)
    maxfx=max(data_fx)
    length_x=len(data_x)

    x=[k*(maxx/disx) for k in range(0,disx)]
    y=[k*(maxfx/disy) for k in range(0,disy)]

    A0x=fr.atleast(x,data_x[0],10/maxx)
    B0y=fr.atmost(y,data_fx[0],10/maxfx)

    A0xx, B0yy = np.meshgrid(A0x,B0y)
    datamodelMamd=fr.conjL(A0xx,B0yy)

    for i in range(1, length_x):
      Aix=fr.atleast(x,data_x[i],10/maxx)
      Biy=fr.atmost(y,data_fx[i],10/maxfx)
      Aixx, Biyy = np.meshgrid(Aix,Biy)
      Aixc_LBiy=fr.conjL(Aixx,Biyy)
      datamodelMamd=np.maximum(datamodelMamd,Aixc_LBiy)

    return datamodelMamd

def RulesIntervals(nodes_x: np.ndarray, nodes_fx: np.ndarray,disx:np.integer,disy:np.integer):
    # Implicative modes that use atl and atm intervals over nodes fx
    maxx=max(nodes_x)
    maxfx=max(nodes_fx)
    length_x=len(nodes_x)

    x=[k*(maxx/disx) for k in range(0,disx)]
    y=[k*(maxfx/disy) for k in range(0,disy)]

    A0x=np.minimum(fr.atleast(x,nodes_x[0],10/maxx),fr.atmost(x,nodes_x[1],10/maxx))
    B0y=np.minimum(fr.atleast(y,nodes_fx[0],10/maxfx),fr.atmost(y,nodes_fx[1],10/maxfx))

    A0xx, B0yy = np.meshgrid(A0x,B0y)
    datamodelRules=fr.implL(A0xx,B0yy)

    for i in range(1, (length_x-1)):
      Aix=np.minimum(fr.atleast(x,nodes_x[i],10/maxx),fr.atmost(x,nodes_x[i+1],10/maxx))
      Biy=np.minimum(fr.atleast(y,nodes_fx[i],10/maxfx),fr.atmost(y,nodes_fx[i+1],10/maxfx))
      Aixx, Biyy = np.meshgrid(Aix,Biy)
      Aixi_LBiy=fr.implL(Aixx,Biyy)
      datamodelRules=np.minimum(datamodelRules,Aixi_LBiy)

    return datamodelRules

def RulesATLxATLy(data_x: np.ndarray, data_fx: np.ndarray,disx:np.integer,disy:np.integer):
    # Atleast model
    maxx=max(data_x)
    maxfx=max(data_fx)
    length_x=len(data_x)

    x=[k*(maxx/disx) for k in range(0,disx)]
    y=[k*(maxfx/disy) for k in range(0,disy)]

    A0x=fr.atleast(x,data_x[0],10/maxx)
    B0y=fr.atleast(y,data_fx[0],10/maxfx)

    A0xx, B0yy = np.meshgrid(A0x,B0y)
    datamodelRules=fr.implL(A0xx,B0yy)

    for i in range(1, length_x):
      Aix=fr.atleast(x,data_x[i],10/maxx)
      Biy=fr.atleast(y,data_fx[i],10/maxfx)
      Aixx, Biyy = np.meshgrid(Aix,Biy)
      Aixi_LBiy=fr.implL(Aixx,Biyy)
      datamodelRules=np.minimum(datamodelRules,Aixi_LBiy)

    return datamodelRules

def MamdSim(data_x: np.ndarray, data_fx: np.ndarray,disx:np.integer,disy:np.integer):
    # Mamdani for each data
    maxx=max(data_x)
    maxfx=max(data_fx)
    length_x=len(data_x)

    x=[k*(maxx/disx) for k in range(0,disx)]
    y=[k*(maxfx/disy) for k in range(0,disy)]

    A0x=fr.pointksim(x,10/maxx,data_x[0])
    B0y=fr.pointksim(y,10/maxfx,data_fx[0])

    A0xx, B0yy = np.meshgrid(A0x,B0y)
    datamodelMamd=fr.conjL(A0xx,B0yy)

    for i in range(1, length_x):
      Aix=fr.pointksim(x,10/maxx,data_x[i])
      Biy=fr.pointksim(y,10/maxfx,data_fx[i])
      Aixx, Biyy = np.meshgrid(Aix,Biy)
      Aixc_LBiy=fr.conjL(Aixx,Biyy)
      datamodelMamd=np.maximum(datamodelMamd,Aixc_LBiy)

    return datamodelMamd

def RulesSim(data_x: np.ndarray, data_fx: np.ndarray,disx:np.integer,disy:np.integer):
    # Rules for each data
    maxx=max(data_x)
    maxfx=max(data_fx)
    length_x=len(data_x)

    x=[k*(maxx/disx) for k in range(0,disx)]
    y=[k*(maxfx/disy) for k in range(0,disy)]

    A0x=fr.pointksim(x,10/maxx,data_x[0])
    B0y=fr.pointksim(y,10/maxfx,data_fx[0])

    A0xx, B0yy = np.meshgrid(A0x,B0y)
    datamodelRules=fr.implL(A0xx,B0yy)

    for i in range(1, length_x):
      Aix=fr.pointksim(x,10/maxx,data_x[i])
      Biy=fr.pointksim(y,10/maxfx,data_fx[i])
      Aixx, Biyy = np.meshgrid(Aix,Biy)
      Aixi_LBiy=fr.implL(Aixx,Biyy)
      datamodelRules=np.minimum(datamodelRules,Aixi_LBiy)

    return datamodelRules

    # SURFACE + SCATTER PLOT
def ShowModelwithData(set_x, set_y, dataset, scatter_x,scatter_fx,scatter_z, name_of_plot="Graf"):
    # Scaling
    x_size = len(dataset[0])
    y_size = len(dataset)

    if x_size > y_size:
        x_scale = 1
        y_scale = y_size / x_size
    else:
        x_scale = x_size / y_size
        y_scale = 1

    if scatter_x is not None:
        fig = go.Figure(
            data=[
                go.Surface(
                    x=set_x,
                    y=set_y,
                    z=dataset,
                    # colorscale='peach',
                    opacity=0.99
                ),
                go.Scatter3d(
                    x=scatter_x,
                    y=scatter_fx,
                    z=scatter_z,
                    mode="markers",
                    marker=dict(
                        size=2,
                        # symbol="x",
                        # color=scatter_dataset["z"],  # set color to an array/list of desired values
                        color="black",
                        # colorscale='teal',  # choose a colorscale
                        # colorscale='blugrn',  # choose a colorscale
                        opacity=0.99
                    )
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Surface(
                    x=set_x,
                    y=set_y,
                    z=dataset,
                    # colorscale='peach',
                    opacity=0.99
                )
            ]
        )

    fig.update_layout(
        title=name_of_plot,
        scene={
            'xaxis_title': 'X',
            'yaxis_title': 'Y',
            'zaxis_title': 'Z',
            'camera_eye': {"x": 0, "y": 0, "z": 2},
            "aspectratio": {"x": x_scale, "y": y_scale, "z": 0.5},
            "zaxis": dict(nticks=5, range=[0, 1])

        }
    )

    fig.show()
    return fig

