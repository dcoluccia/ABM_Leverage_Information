import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import main

n_investors = 20
n_corporates = 10
n_equity = [1 for j in range(n_corporates)]
initial_value = [1 for j in range(n_corporates)]
w = 5
time_to_maturity = [150 for j in range(n_corporates)]
discount_rate = 0.2
beta = 0.8
delta = [np.random.triangular(0.4, 0.5, 0.6) for j in range(n_investors)] #0.4, 0.5, 0.6 #0.1,0.2,0.3
mi = [np.random.triangular(0.15,0.2,0.25) for j in range(n_investors)] #0.15,0.2,0.25 #0.015,0.020,0.025
sigma = [np.random.triangular(0.2, 0.3, 0.4) for j in range(n_investors)] #0.2, 0.3, 0.4 #0.1,0.15,0.20
gamma = 0.6
eta = [0 for j in range(n_investors)] #0
noise = 0
rho = 0.1
iterations = 50
face_value = [0 for j in range(n_corporates)]

time_periods = [i+1 for i in range(iterations-1)]

A = []
P = []
R = []
Debt = []
Delta = []
E = []
Model = main.Equity_Debt(n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations)
Model.populate_returns()
Model.populate_endowments_holdings()
for index in range(2):
    if index == 0: #gamma=1
        a = [];p=[];r=[];debt=[];e=[]
        for t in time_periods:
            #Model.temporary_shock()
            Model.permanent_shock(t)
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            a.append(np.average(Model.assets[t]))
            p.append(np.average(Model.P[t+1]))
            r.append(np.average(Model.true_returns))
            debt.append(Model.debt_all_shocked)
            e.append(Model.aggregate_endowment)
        A.append(a);P.append(p);R.append(r);Debt.append(debt);E.append(e)
    else: #gamma=0.7
        a = [];p=[];r=[];debt=[];e=[]
        for t in time_periods:
            Model.adjust(t)
            #Model.adjust_gamma(t)
            #Model.temporary_shock()
            Model.permanent_shock(t)
            Model.adjust_face_values(t)
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            a.append(np.average(Model.assets[t]))
            p.append(np.average(Model.P[t+1]))
            r.append(np.average(Model.true_returns))
            debt.append(Model.debt_all_shocked)
            e.append(Model.aggregate_endowment)
        A.append(a);P.append(p);R.append(r);Debt.append(debt);E.append(e)
A = np.array(A); P = np.array(P); R = np.array(R); Debt = np.array(Debt)
#############
Delta = []
d_1 = []
d_2 = []
for t in range(iterations-1):
    d_1.append((P[0][t]/R[0][t])*100)
    d_2.append((P[1][t]/R[1][t])*100)
Delta.append(d_1);Delta.append(d_2);Delta = np.array(Delta)
##############
Diff = []
diff_1 = []
diff_2 = []
for t in range(iterations-1):
    diff_1.append(np.divide(A[0][t],P[0][t]*np.average(n_equity)+Debt[0][t]))
    diff_2.append(np.divide(A[1][t],P[1][t]*np.average(n_equity)+Debt[1][t]))
Diff.append(diff_1);Diff.append(diff_2);Diff = np.array(Diff)
###############
Leverage = []
lev_1 = []
lev_2 = []
for t in range(iterations-1):
    lev_1.append(A[0][t]/(P[0][t]*np.average(n_equity)))
    lev_2.append(A[1][t]/(P[1][t]*np.average(n_equity)))
Leverage.append(lev_1);Leverage.append(lev_2);Leverage = np.array(Leverage)
###############
Ratio_endowment = []
for t in range(iterations-1):
    Ratio_endowment.append(E[0][t]/E[1][t])
Ratio_endowment = np.array(Ratio_endowment)
###############
Ratio_liabilities = []
L_debt = []
L_no_Debt = []
ratio = []
for t in range(iterations-1):
    l_debt = P[1][t]*np.average(n_equity)+Debt[1][t]
    l_no_Debt = P[0][t]*np.average(n_equity)+Debt[0][t]
    L_debt.append(l_debt)
    L_no_Debt.append(l_no_Debt)
    ratio.append(l_debt/l_no_Debt)
Ratio_liabilities.append(L_debt);Ratio_liabilities.append(L_no_Debt);Ratio_liabilities.append(ratio)
###############
# Leverage
d_LEV = {'A/E: debt': Leverage[1],'A/E: no debt':Leverage[0]}
# Confront shock liabilities-assets
# d_PI = {'A/L: no debt':Diff[0]}
# d_II = {'A/L: debt': Diff[1]}
# log values
# d_PI = {'log-Value Assets':np.log(A[0]),'log-Price':np.log(P[0]),'log-Roe':np.log(R[0]),'log-Value Debt':np.log(Debt[0]),'P-R':Delta[0]}
# d_II = {'log-Value Assets':np.log(A[1]),'log-Price':np.log(P[1]),'log-Roe':np.log(R[1]),'log-Value Debt':np.log(Debt[1]),'P-R':Delta[1]}
# Endowments
d1 = {'Y - No Debt':np.log(E[0]),'Y - Debt':np.log(E[1]),'Ratio':Ratio_endowment}
# Liabilities
d = {'L-Ratio':Ratio_liabilities[2],'Growth Ratio':np.append(np.diff(np.log(Ratio_liabilities[2])),0),'L : no debt':Ratio_liabilities[1],'L : debt':Ratio_liabilities[0]}
# level values
d_PI = {'Value Assets':A[0],'Price':P[0],'Roe':R[0],'Value Debt':Debt[0],'P-R':Delta[0]}
d_II = {'Value Assets':A[1],'Price':P[1],'Roe':R[1],'Value Debt':Debt[1],'P-R':Delta[1]}
df_PI = pd.DataFrame(d_PI)
df_II = pd.DataFrame(d_II)
df_LEV = pd.DataFrame(d_LEV)
df = pd.DataFrame(d)
df1 = pd.DataFrame(d1)
df_PI.plot(title='Equity + No Debt: Imperfect Information',subplots=True)
df_II.plot(title='Equity + Debt: Imperfect Information',subplots=True)
df_LEV.plot(title='Leverage: Imperfect Information',subplots=False)
df.plot(title='Liabilities: Imperfect Information')
df1.plot(title='Aggregate Endowment: Imperfect Information')
plt.show()
