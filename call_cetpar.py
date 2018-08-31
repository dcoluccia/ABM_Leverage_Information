import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#plt.style.use('ggplot')
import pandas as pd
from scipy.interpolate import griddata
import main
import copy
import itertools
### CALCULATE SHOCK PERSISTENCE ###
def shock_persistence(list,t_0,t_shock):
    persistence = 0
    for t in range(len(list)):
        if t >= t_shock:
            spread = np.divide(list[t],list[t_0])*100
            if spread > 110 or spread < 90:
                persistence += 1
            else:
                break
    return persistence
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
face_value = [0.8*initial_value[j] for j in range(n_corporates)]#[0 for j in range(M)] #
sim = 15
####################
Spread = [];Leverage = []
Spread_width = [];Spread_persistence = []
Leverage_width = [];Leverage_persistence = []
time_periods = [i+1 for i in range(iterations-1)]
list_delta = np.linspace(0.11,0.89,num=sim)
list_b = np.linspace(0.1,0.9,num=sim)
list_i = list(range(sim))
list_j = list(range(sim))
####################
Model = main.Equity_Debt(n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations,sim)
Model.populate_returns()
Model.populate_endowments_holdings()
Model.temporary_shock()
ggrid = []
for i in list_i:
    for j in list_j:
        #list to be empty
        spread = []; leverage = []
        p = []; r = []; a = []
        #simulation
        for t in time_periods:
            Model.adjust(t)
            Model.adjust_parspace(i,j)
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
        #evaluate measures
        spread_width = np.divide(spread[25] - spread[24],spread[24])**2 + np.divide(spread[26] - spread[25],spread[25])**2 + np.divide(spread[27] - spread[26],spread[26])**2
        spread_persistence = shock_persistence(spread,20,25)
        leverage_width = np.divide(leverage[25]-leverage[24],leverage[24])**2 + np.divide(leverage[26]-leverage[25],leverage[25])**2 + np.divide(leverage[27]-leverage[26],leverage[26])**2
        leverage_persistence = shock_persistence(leverage,20,25)
        #append measures and series to panel-wide
        Spread.append(spread);Leverage.append(Leverage)
        Spread_width.append(spread_width);Spread_persistence.append(spread_persistence)
        Leverage_width.append(leverage_width);Leverage_persistence.append(leverage_persistence)
        ggrid.append((list_delta[i],list_b[j]))
##############
grid = list(itertools.product(list_delta, list_b))
assert grid == ggrid
list_delta = [grid[i][0] for i in range(len(grid))]
list_b = [grid[i][1] for i in range(len(grid))]
##############
dict_1 = {'list_delta':list_delta, 'list_b':list_b, 'Spread_width':Spread_width, 'Spread_persistence':Spread_persistence, 'Leverage_width':Leverage_width, 'Leverage_persistence':Leverage_persistence}
df = pd.DataFrame(dict_1, index = range(len(dict_1['list_delta'])))
#
list_delta1 = np.linspace(df['list_delta'].min(), df['list_delta'].max(), len(df['list_delta'].unique()))
list_b1 = np.linspace(df['list_b'].min(), df['list_b'].max(), len(df['list_b'].unique()))
list_delta2, list_b2 = np.meshgrid(list_delta1,list_b1)
#
Spread_width2 = griddata((df['list_delta'],df['list_b']),df['Spread_width'], (list_delta2,list_b2), method = 'cubic')
Spread_persistence2 = griddata((df['list_delta'],df['list_b']),df['Spread_persistence'], (list_delta2,list_b2), method = 'cubic')
Leverage_width2 = griddata((df['list_delta'],df['list_b']),df['Leverage_width'], (list_delta2,list_b2), method = 'cubic')
Leverage_persistence2 = griddata((df['list_delta'],df['list_b']),df['Leverage_persistence'], (list_delta2,list_b2), method = 'cubic')
##############
fig1 = plt.figure()
ax1 = fig1.gca(projection = '3d')
surf1 = ax1.plot_surface(list_delta2, list_b2, Spread_width2, cmap=cm.coolwarm, rstride = 1, cstride = 1)#, linewidth = 0.0)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax1.set_xlabel('Delta')
ax1.set_ylabel('Leverage')
ax1.set_zlabel('Price shock amplification')
fig2 = plt.figure()
ax2 = fig2.gca(projection = '3d')
surf2 = ax2.plot_surface(list_delta2, list_b2, Spread_persistence2, cmap=cm.coolwarm, rstride = 1, cstride = 1)#, linewidth = 0.0)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax2.set_xlabel('Delta')
ax2.set_ylabel('Leverage')
ax2.set_zlabel('Price shock persistence')
fig3 = plt.figure()
ax3 = fig3.gca(projection = '3d')
surf3 = ax3.plot_surface(list_delta2, list_b2, Leverage_width2, cmap=cm.coolwarm, rstride = 1, cstride = 1)#, linewidth = 0.0)
ax3.zaxis.set_major_locator(LinearLocator(10))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax3.set_xlabel('Delta')
ax3.set_ylabel('Leverage')
ax3.set_zlabel('Leverage shock amplification')
fig4 = plt.figure()
ax4 = fig4.gca(projection = '3d')
surf4 = ax4.plot_surface(list_delta2, list_b2, Leverage_persistence2, cmap=cm.coolwarm, rstride = 1, cstride = 1)#, linewidth = 0.0)
ax4.zaxis.set_major_locator(LinearLocator(10))
ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax4.set_xlabel('Delta')
ax4.set_ylabel('Leverage')
ax4.set_zlabel('Leverage shock persistence')
plt.show()
