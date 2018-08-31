import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import main
import copy
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
sim = 1
####################
time_periods = [i+1 for i in range(iterations-1)]
R = [] #returns
P = [] #prices
A = [] #assets
####################
Model = main.Equity_Debt(n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations,sim)
Model.populate_returns()
Model.populate_endowments_holdings()
Model.temporary_shock()
for index in range(3):
    if index == 0: #mid chartist
        p = []; r = []; a = []
        for t in time_periods:
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            r.append(np.average(Model.true_returns));p.append(np.average(Model.P[t+1]));a.append(np.average(Model.assets[t]))
        R.append(r);P.append(p);A.append(a)
    elif index == 1: #low chartist
        p = []; r = []; a = []
        for t in time_periods:
            Model.adjust(t)
            Model.adjust_debt(index)
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            r.append(np.average(Model.true_returns));p.append(np.average(Model.P[t+1]));a.append(np.average(Model.assets[t]))
        R.append(r);P.append(p);A.append(a)
    else: #high chartist
        p = []; r = []; a = []
        for t in time_periods:
            Model.adjust(t)
            Model.adjust_debt(index)
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            r.append(np.average(Model.true_returns));p.append(np.average(Model.P[t+1]));a.append(np.average(Model.assets[t]))
        R.append(r);P.append(p);A.append(a)
###############
Spread = []
s_1 = []
s_2 = []
s_3 = []
for t in range(59):
    s_1.append(P[0][t]/R[0][t]*100)
    s_2.append(P[1][t]/R[1][t]*100)
    s_3.append(P[2][t]/R[2][t]*100)
Spread.append(s_1);Spread.append(s_2);Spread.append(s_3)
###############
Leverage = []
l_1 = []
l_2 = []
l_3 = []
for t in range(59):
    l_1.append(A[0][t]/(P[0][t]*np.average(n_equity)))
    l_2.append(A[1][t]/(P[1][t]*np.average(n_equity)))
    l_3.append(A[2][t]/(P[2][t]*np.average(n_equity)))
Leverage.append(l_1);Leverage.append(l_2);Leverage.append(l_3)
###############
s = copy.deepcopy(Spread); l = copy.deepcopy(Leverage)
t = 10
for i in range(3):
    Spread[i] = s[i][t:]
    Leverage[i] = l[i][t:]
##############
plt.figure(1)
plt.plot(Spread[0],'b',label='lev = 0.8');plt.plot(Spread[2],'r',label='lev = 0.5');plt.plot(Spread[1],'g',label='lev = 0.2')
plt.legend();plt.title('Price spreads')
plt.figure(2)
plt.plot(Leverage[0],'b',label='lev = 0.8');plt.plot(Leverage[2],'r',label='lev = 0.5');plt.plot(Leverage[1],'g',label='lev = 0.2')
plt.legend();plt.title('Leverage')
plt.show()
