import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import gaussian_kde
plt.style.use('ggplot')
import pandas as pd
from scipy.interpolate import griddata
import main
import copy
import itertools
### CALIBRATION ###
n_investors = 20
n_corporates = 10
n_equity = [1 for j in range(n_corporates)]
initial_value = [1 for j in range(n_corporates)]
w = 5
time_to_maturity = [150 for j in range(n_corporates)]
discount_rate = 0.2
beta = 0.8
delta = [np.random.triangular(0.4, 0.5, 0.6) for j in range(n_investors)] #0.4, 0.5, 0.6 #0.1,0.2,0.3
mi = [np.random.triangular(0.015,0.02,0.025) for j in range(n_investors)] #0.015,0.020,0.025 #0.05, 0.06, 0.07
sigma = [np.random.triangular(0.1,0.15,0.20) for j in range(n_investors)] #0.01,0.015,0.020
gamma = 0.6
eta = [0 for j in range(n_investors)] #[np.random.triangular(0.3,0.5,0.7) for j in range(n_investors)]
noise = 0
rho = 0.1
iterations = 60
face_value = [0.85*initial_value[j] for j in range(n_corporates)]#[0 for j in range(M)] #
sim = 100
####################
Spread = [];Leverage = []
time_periods = [i+1 for i in range(iterations-1)]
####################
for iteration in range(sim):
    #
    r = []; p = []; a = []
    spread = []; leverage = []
    #
    Model = main.Equity_Debt(n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations,sim)
    Model.populate_returns()
    Model.populate_endowments_holdings()
    Model.temporary_shock()
    for t in time_periods:
        Model.adjust(t)
        #Model.adjust_parspace(i,j)
        Model.compute_weight_matrix(t)
        Model.market(t)
        Model.update_endowments(t)
        r.append(np.average(Model.true_returns));p.append(np.average(Model.P[t+1]));a.append(np.average(Model.assets[t]))
    #delete transient
    cut_time = 10
    r_copy = copy.deepcopy(r);p_copy = copy.deepcopy(p);a_copy = copy.deepcopy(a)
    r = r_copy[cut_time:];p = p_copy[cut_time:];a = a_copy[cut_time:]
    #get interesting series
    for t in range(iterations-cut_time-1):
        spread.append(p[t]/r[t]*100)
        leverage.append(a[t]/(p[t]*np.average(n_equity)))
    Spread.append(spread);Leverage.append(leverage)
##############
x = []
for j in range(iterations-cut_time-1):
    for i in range(sim):
        x.append(j)
y_spread = []
y_leverage = []
for i in range(sim):
    for j in range(iterations-cut_time-1):
        y_spread.append(Spread[i][j])
        y_leverage.append(Leverage[i][j])
z_spread = gaussian_kde(y_spread)(y_spread); z_leverage = gaussian_kde(y_leverage)(y_leverage)
idx_spread = z_spread.argsort(); idx_leverage = z_leverage.argsort()
x_spread = copy.deepcopy(x); x_leverage = copy.deepcopy(x)
# x_spread, y_spread, z_spread = x_spread[idx_spread], y_spread[idx_spread], z_spread[idx_spread]
# x_leverage, y_leverage, z_leverage = x_leverage[idx_leverage], y_leverage[idx_leverage], z_leverage[idx_leverage]
##############
fig1, ax1 = plt.subplots()
cax1 = ax1.scatter(x_spread, y_spread, c=z_spread, s=30, edgecolor='')
ax1.set_title('Price Spread')
fig1.colorbar(cax1)
fig2, ax2 = plt.subplots()
cax2 = ax2.scatter(x_leverage, y_leverage, c=z_leverage, s=30, edgecolor = '')
ax2.set_title('Leverage')
fig2.colorbar(cax2)
plt.show()
