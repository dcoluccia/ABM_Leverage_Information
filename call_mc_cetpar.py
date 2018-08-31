import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import colors as mcolors
#plt.style.use('ggplot')
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
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
sim = 5
simulations = 50
####################
Spread = [];Leverage = []
Spread_width = [];Spread_persistence = []
mc_Spread_width = []; mc_Spread_persistence = []
mc_Leverage_width = []; mc_Leverage_persistence = []
Leverage_width = [];Leverage_persistence = []
time_periods = [i+1 for i in range(iterations-1)]
list_delta = np.linspace(0.11,0.89,num=sim)
list_b = np.linspace(0.1,0.9,num=sim)
list_i = list(range(sim))
list_j = list(range(sim))
####################
for simul in range(simulations):
    Spread_width = []; Spread_persistence = []
    Leverage_width = []; Leverage_persistence = []
    Model = main.Equity_Debt(n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations,sim)
    Model.populate_returns()
    Model.populate_endowments_holdings()
    Model.temporary_shock()
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
    mc_Spread_width.append(Spread_width); mc_Spread_persistence.append(Spread_persistence)
    mc_Leverage_width.append(Leverage_width);mc_Leverage_persistence.append(Leverage_persistence)
##############
grid = list(itertools.product(list_delta, list_b))
list_delta = [grid[i][0] for i in range(len(grid))]
list_b = [grid[i][1] for i in range(len(grid))]
#############
within_Spread_width = [[] for i in range(len(grid))]
within_Spread_persistence = [[] for i in range(len(grid))]
within_Leverage_width = [[] for i in range(len(grid))]
within_Leverage_persistence = [[] for i in range(len(grid))]
for i in range(len(mc_Spread_width)):
    for j in range(len(mc_Spread_width[i])):
        within_Spread_width[j].append(mc_Spread_width[i][j])
        within_Spread_persistence[j].append(mc_Spread_persistence[i][j])
        within_Leverage_width[j].append(mc_Leverage_width[i][j])
        within_Leverage_persistence[j].append(mc_Leverage_persistence[i][j])
###############
# m = [1000 for i in range(len(grid))]
# M = [-1000 for i in range(len(grid))]
# for i in range(len(grid)):
#     if min(within_Spread_width[i]) < m[i] and max(within_Spread_width[i]) > M[i]:
#         m[i] = min(within_Spread_width[i])
#         M[i] = max(within_Spread_width[i])
#     elif min(within_Spread_width[i]) < m[i] and max(within_Spread_width[i]) < M[i]:
#         m[i] = min(within_Spread_width[i])
#     elif min(within_Spread_width[i]) > m[i] and max(within_Spread_width[i]) < M[i]:
#         M[i] = max(within_Spread_width[i])
#     else:
#         pass
# m_SpreadWidth = m
# M_SpreadWidth = M
# PriceSpread_space = [np.linspace(m_SpreadWidth[i],M_SpreadWidth[i],50) for i in range(len((grid)))]
# #
# m = [1000 for i in range(len(grid))]
# M = [-1000 for i in range(len(grid))]
# for i in range(len(grid)):
#     if min(within_Leverage_width[i]) < m[i] and max(within_Leverage_width[i]) > M[i]:
#         m[i] = min(within_Leverage_width[i])
#         M[i] = max(within_Leverage_width[i])
#     elif min(within_Leverage_width[i]) < m[i] and max(within_Leverage_width[i]) < M[i]:
#         m[i] = min(within_Leverage_width[i])
#     elif min(within_Leverage_width[i]) > m[i] and max(within_Leverage_width[i]) < M[i]:
#         M[i] = max(within_Leverage_width[i])
#     else:
#         pass
# m_LeverageWidth = m
# M_LeverageWidth = M
# LeverageSpread_space = [np.linspace(m_LeverageWidth[i],M_LeverageWidth[i],50) for i in range(len((grid)))]
# #
# m = [1000 for i in range(len(grid))]
# M = [-1000 for i in range(len(grid))]
# for i in range(len(grid)):
#     if min(within_Spread_persistence[i]) < m[i] and max(within_Spread_persistence[i]) > M[i]:
#         m[i] = min(within_Spread_persistence[i])
#         M[i] = max(within_Spread_persistence[i])
#     elif min(within_Spread_persistence[i]) < m[i] and max(within_Spread_persistence[i]) < M[i]:
#         m[i] = min(within_Spread_persistence[i])
#     elif min(within_Spread_persistence[i]) > m[i] and max(within_Spread_persistence[i]) < M[i]:
#         M[i] = max(within_Spread_persistence[i])
#     else:
#         pass
# m_SpreadPersistence = m
# M_SpreadPersistence = M
# PricePersistence_space = [np.linspace(m_SpreadPersistence[i],M_SpreadPersistence[i],50) for i in range(len((grid)))]
# #
# m = [1000 for i in range(len(grid))]
# M = [-1000 for i in range(len(grid))]
# for i in range(len(grid)):
#     if min(within_Leverage_persistence[i]) < m[i] and max(within_Leverage_persistence[i]) > M[i]:
#         m[i] = min(within_Leverage_persistence[i])
#         M[i] = max(within_Leverage_persistence[i])
#     elif min(within_Leverage_persistence[i]) < m[i] and max(within_Leverage_persistence[i]) < M[i]:
#         m[i] = min(within_Leverage_persistence[i])
#     elif min(within_Leverage_persistence[i]) > m[i] and max(within_Leverage_persistence[i]) < M[i]:
#         M[i] = max(within_Leverage_persistence[i])
#     else:
#         pass
# m_LeveragePersistence = m
# M_LeveragePersistence = M
# LeveragePersistence_space = [np.linspace(m_LeveragePersistence[i],M_LeveragePersistence[i],50) for i in range(len((grid)))]
# ###############
# mc_kde_spread_width = []
# mc_kde_spread_persistence = []
# mc_kde_leverage_width = []
# mc_kde_leverage_persistence = []
# for i in range(len(grid)):
#     kde_spread_width = gaussian_kde(within_Spread_width[i])
#     kde_leverage_width = gaussian_kde(within_Leverage_width[i])
#     if np.std(within_Spread_persistence[i]) != 0:
#         kde_spread_persistence = gaussian_kde(within_Spread_persistence[i])
#     else:
#         for j in range(len(within_Spread_persistence[i])):
#             within_Spread_persistence[i][j] += np.random.normal(0,0.01)
#         kde_spread_persistence = gaussian_kde(within_Spread_persistence[i])
#     if np.std(within_Leverage_persistence[i]) != 0:
#         kde_leverage_persistence = gaussian_kde(within_Leverage_persistence[i])
#     else:
#         for j in range(len(within_Leverage_persistence[i])):
#             within_Leverage_persistence[i][j] += np.random.normal(0,0.01)
#         kde_leverage_persistence = gaussian_kde(within_Leverage_persistence[i])
#     mc_kde_spread_width.append(kde_spread_width)
#     mc_kde_spread_persistence.append(kde_spread_persistence)
#     mc_kde_leverage_width.append(kde_leverage_width)
#     mc_kde_leverage_persistence.append(kde_leverage_persistence)
##############
within_Spread_width_sd = [np.std(within_Spread_width[i]) for i in range(len(grid))]
within_Spread_persistence_sd = [np.std(within_Spread_persistence[i]) for i in range(len(grid))]
within_Leverage_width_sd = [np.std(within_Leverage_width[i]) for i in range(len(grid))]
within_Leverage_persistence_sd = [np.std(within_Leverage_persistence[i]) for i in range(len(grid))]
within_Spread_width_mean = [np.mean(within_Spread_width[i]) for i in range(len(grid))]
within_Spread_persistence_mean = [np.mean(within_Spread_persistence[i]) for i in range(len(grid))]
within_Leverage_width_mean = [np.mean(within_Leverage_width[i]) for i in range(len(grid))]
within_Leverage_persistence_mean = [np.mean(within_Leverage_persistence[i]) for i in range(len(grid))]
#
dict_price = {'list_delta':list_delta, 'list_b':list_b, 'within_Spread_width_mean':within_Spread_width_mean, 'within_Spread_width_sd':within_Spread_width_sd, 'within_Spread_persistence_mean':within_Spread_persistence_mean, 'within_Spread_persistence_sd':within_Spread_persistence_sd}
dict_leverage = {'list_delta':list_delta, 'list_b':list_b, 'within_Leverage_width_mean':within_Leverage_width_mean, 'within_Leverage_width_sd':within_Leverage_width_sd, 'within_Leverage_persistence_mean':within_Leverage_persistence_mean, 'within_Leverage_persistence_sd':within_Leverage_persistence_sd}
#
df_price = pd.DataFrame(dict_price, index = range(len(dict_price['list_delta'])))
df_leverage = pd.DataFrame(dict_leverage, index = range(len(dict_price['list_delta'])))
#
##############
fig1 = plt.figure()
ax1 = fig1.add_subplot(111,projection = '3d')
for i in range(len(df_price['list_delta'])):
    ax1.plot([df_price['list_delta'][i],df_price['list_delta'][i]],[df_price['list_b'][i],df_price['list_b'][i]],[df_price['within_Spread_width_mean'][i] + df_price['within_Spread_width_sd'][i],df_price['within_Spread_width_mean'][i] - df_price['within_Spread_width_sd'][i]],marker = "_", color = 'royalblue')
ax1.scatter(df_price['list_delta'],df_price['list_b'],df_price['within_Spread_width_mean'], color = 'royalblue')
ax1.set_xlabel('Delta')
ax1.set_ylabel('Debt')
ax1.set_zlabel('Price spread shock')
#
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection = '3d')
for i in range(len(df_price['list_delta'])):
    ax2.plot([df_price['list_delta'][i],df_price['list_delta'][i]],[df_price['list_b'][i],df_price['list_b'][i]],[df_price['within_Spread_persistence_mean'][i] + df_price['within_Spread_persistence_sd'][i],df_price['within_Spread_persistence_mean'][i] - df_price['within_Spread_persistence_sd'][i]],marker = "_", color = 'firebrick')
ax2.scatter(df_price['list_delta'],df_price['list_b'],df_price['within_Spread_persistence_mean'], color = 'firebrick')
ax2.set_xlabel('Delta')
ax2.set_ylabel('Debt')
ax2.set_zlabel('Price shock persistence')
#
fig3 = plt.figure()
ax3 = fig3.add_subplot(111,projection = '3d')
for i in range(len(df_leverage['list_delta'])):
    ax3.plot([df_leverage['list_delta'][i],df_leverage['list_delta'][i]],[df_leverage['list_b'][i],df_leverage['list_b'][i]],[df_leverage['within_Leverage_width_mean'][i] + df_leverage['within_Leverage_width_sd'][i],df_leverage['within_Leverage_width_mean'][i] - df_leverage['within_Leverage_width_sd'][i]],marker = "_", color = 'royalblue')
ax3.scatter(df_leverage['list_delta'],df_leverage['list_b'],df_leverage['within_Leverage_width_mean'], color = 'royalblue')
ax3.set_xlabel('Delta')
ax3.set_ylabel('Debt')
ax3.set_zlabel('Leverage shock amplification')
#
fig4 = plt.figure()
ax4 = fig4.add_subplot(111,projection = '3d')
for i in range(len(df_leverage['list_delta'])):
    ax4.plot([df_leverage['list_delta'][i],df_leverage['list_delta'][i]],[df_leverage['list_b'][i],df_leverage['list_b'][i]],[df_leverage['within_Leverage_persistence_mean'][i] + df_leverage['within_Leverage_persistence_sd'][i],df_leverage['within_Leverage_persistence_mean'][i] - df_leverage['within_Leverage_persistence_sd'][i]],marker = "_", color = 'firebrick')
ax4.scatter(df_leverage['list_delta'],df_leverage['list_b'],df_leverage['within_Leverage_persistence_mean'], color = 'firebrick')
ax4.set_xlabel('Delta')
ax4.set_ylabel('Debt')
ax4.set_zlabel('Leverage shock persistence')
plt.show()
# new_grid = np.around(grid,decimals=1)
# fig1 = plt.figure ()
# ax1 = fig1.add_subplot(111)
# for i in range(len(grid)):
#     ax1.plot(PriceSpread_space[i],mc_kde_spread_width[i](PriceSpread_space[i]), label = new_grid[i])
# plt.legend()
# plt.ylabel('Probability density');plt.xlabel('(log) Price spread')
# plt.yscale('log');plt.xscale('log')
# fig2 = plt.figure ()
# ax2 = fig2.add_subplot(111)
# for i in range(len(grid)):
#     ax2.plot(LeverageSpread_space[i],mc_kde_leverage_width[i](LeverageSpread_space[i]), label = new_grid[i])
# plt.legend()
# plt.ylabel('Probability density');plt.xlabel('(log) Leverage spread')
# plt.yscale('log');plt.xscale('log')
# fig3 = plt.figure ()
# ax3 = fig3.add_subplot(111)
# for i in range(len(grid)):
#     ax3.plot(PricePersistence_space[i],mc_kde_spread_persistence[i](PricePersistence_space[i]), label = new_grid[i])
# plt.legend()
# plt.ylabel('Probability density');plt.xlabel('(log) Price spread persistence')
# plt.yscale('log');plt.xscale('log')
# fig4 = plt.figure ()
# ax4 = fig4.add_subplot(111)
# for i in range(len(grid)):
#     ax4.plot(LeveragePersistence_space[i],mc_kde_leverage_persistence[i](LeveragePersistence_space[i]), label = new_grid[i])
# plt.legend()
# plt.ylabel('Probability density');plt.xlabel('(log) Leverage spread persistence')
# plt.yscale('log');plt.xscale('log')
# plt.show()
