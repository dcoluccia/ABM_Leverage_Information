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
mi = [np.random.triangular(0.015,0.02,0.025) for j in range(n_investors)] #0.015,0.020,0.025
sigma = [np.random.triangular(0.01,0.015,0.020) for j in range(n_investors)] #0.01,0.015,0.020
gamma = 0.6
eta = [0.2 for j in range(n_investors)] #[np.random.triangular(0.3,0.5,0.7) for j in range(n_investors)]
noise = 0
rho = 0.1
iterations = 50
face_value = [0 for j in range(n_corporates)]#[0 for j in range(M)] #

time_periods = [i+1 for i in range(iterations-1)]

A_shocked = []; A_unshocked = []
P_shocked = []; P_unshocked = []
R_shocked = []; R_unshocked = []
Debt_shocked = []; Debt_unshocked = []
Delta_shocked = []; Delta_unshocked = []
Model = main.Equity_Debt(n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations)
Model.populate_returns()
Model.populate_endowments_holdings()
for index in range(2):
    if index == 0: #gamma=1
        a_shocked = [];a_unshocked = []; p=[];r=[]; p_shocked = []; p_unshocked = []; r_shocked = []; r_unshocked = []; debt_shocked = []; debt_unshocked = []
        for t in time_periods:
            Model.onecorporate_temporary_shock()
            #Model.permanent_shock(t)
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            a_shocked.append(Model.assets[t][0]);a_unshocked.append(Model.assets[t][4])
            p_shocked.append(Model.P[t+1][0]);p_unshocked.append(Model.P[t+1][4])
            r_shocked.append(Model.true_returns[0]);r_unshocked.append(Model.true_returns[4])
            debt_shocked.append(Model.debt_one_shocked[0]);debt_unshocked.append(Model.debt_one_shocked[4])
        A_shocked.append(a_shocked);A_unshocked.append(a_unshocked)
        P_shocked.append(p_shocked);P_unshocked.append(p_unshocked)
        R_shocked.append(r_shocked);R_unshocked.append(r_unshocked)
        Debt_shocked.append(debt_shocked);Debt_unshocked.append(debt_unshocked)
    else: #gamma=0.7
        a_shocked = [];a_unshocked = []; p=[];r=[]; p_shocked = []; p_unshocked = []; r_shocked = []; r_unshocked = []; debt_shocked = []; debt_unshocked = []
        for t in time_periods:
            #Model.adjust_gamma(t)
            Model.adjust(t)
            Model.adjust_face_values(t)
            Model.onecorporate_temporary_shock()
            #Model.permanent_shock(t)
            Model.compute_weight_matrix(t)
            Model.market(t)
            Model.update_endowments(t)
            a_shocked.append(Model.assets[t][0]);a_unshocked.append(Model.assets[t][4])
            p_shocked.append(Model.P[t+1][0]);p_unshocked.append(Model.P[t+1][4])
            r_shocked.append(Model.true_returns[0]);r_unshocked.append(Model.true_returns[4])
            debt_shocked.append(Model.debt_one_shocked[0]);debt_unshocked.append(Model.debt_one_shocked[4])
        A_shocked.append(a_shocked);A_unshocked.append(a_unshocked)
        P_shocked.append(p_shocked);P_unshocked.append(p_unshocked)
        R_shocked.append(r_shocked);R_unshocked.append(r_unshocked)
        Debt_shocked.append(debt_shocked);Debt_unshocked.append(debt_unshocked)
A_shocked = np.array(A_shocked); A_unshocked = np.array(A_unshocked)
P_shocked = np.array(P_shocked); P_unshocked = np.array(P_unshocked)
R_shocked = np.array(R_shocked); R_unshocked = np.array(R_unshocked)
Debt_shocked = np.array(Debt_shocked); Debt_unshocked = np.array(Debt_unshocked)
#############
Delta_shocked = []
d_1 = []
d_2 = []
for t in range(iterations-1):
    d_1.append(P_shocked[0][t]/R_shocked[0][t]*100)
    d_2.append(P_shocked[1][t]/R_shocked[1][t]*100)
Delta_shocked.append(d_1);Delta_shocked.append(d_2);Delta_shocked = np.array(Delta_shocked)
#
Delta_unshocked = []
d_1 = []
d_2 = []
for t in range(iterations-1):
    d_1.append(P_unshocked[0][t]/R_unshocked[0][t]*100)
    d_2.append(P_unshocked[1][t]/R_unshocked[1][t]*100)
Delta_unshocked.append(d_1);Delta_unshocked.append(d_2);Delta_unshocked = np.array(Delta_unshocked)
##############
Diff_shocked = []
diff_1 = []
diff_2 = []
for t in range(iterations-1):
    diff_1.append(np.divide(A_shocked[0][t],P_shocked[0][t]*np.average(n_equity)+Debt_shocked[0][t]))
    diff_2.append(np.divide(A_shocked[1][t],P_shocked[1][t]*np.average(n_equity)+Debt_shocked[1][t]))
Diff_shocked.append(diff_1);Diff_shocked.append(diff_2);Diff_shocked = np.array(Diff_shocked)
#
Diff_unshocked = []
diff_1 = []
diff_2 = []
for t in range(iterations-1):
    diff_1.append(np.divide(A_unshocked[0][t],P_unshocked[0][t]*np.average(n_equity)+Debt_unshocked[0][t]))
    diff_2.append(np.divide(A_unshocked[1][t],P_unshocked[1][t]*np.average(n_equity)+Debt_unshocked[1][t]))
Diff_unshocked.append(diff_1);Diff_unshocked.append(diff_2);Diff_unshocked = np.array(Diff_unshocked)
###############
Leverage_shocked = []
lev_1 = []
lev_2 = []
for t in range(iterations-1):
    lev_1.append(A_shocked[0][t]/(P_shocked[0][t]*np.average(n_equity)))
    lev_2.append(A_shocked[1][t]/(P_shocked[1][t]*np.average(n_equity)))
Leverage_shocked.append(lev_1);Leverage_shocked.append(lev_2);Leverage_shocked = np.array(Leverage_shocked)
#
Leverage_unshocked = []
lev_1 = []
lev_2 = []
for t in range(iterations-1):
    lev_1.append(A_unshocked[0][t]/(P_unshocked[0][t]*np.average(n_equity)))
    lev_2.append(A_unshocked[1][t]/(P_unshocked[1][t]*np.average(n_equity)))
Leverage_unshocked.append(lev_1);Leverage_unshocked.append(lev_2);Leverage_unshocked = np.array(Leverage_unshocked)
###############
Ratio_liabilities_shocked = []
L_debt = []
L_no_Debt = []
ratio = []
for t in range(iterations-1):
    l_debt = P_shocked[1][t]*np.average(n_equity)+Debt_shocked[1][t]
    l_no_Debt = P_shocked[0][t]*np.average(n_equity)+Debt_shocked[0][t]
    L_debt.append(l_debt)
    L_no_Debt.append(l_no_Debt)
    ratio.append(l_debt/l_no_Debt)
Ratio_liabilities_shocked.append(L_debt);Ratio_liabilities_shocked.append(L_no_Debt);Ratio_liabilities_shocked.append(ratio)
#
Ratio_liabilities_unshocked = []
L_debt = []
L_no_Debt = []
ratio = []
for t in range(iterations-1):
    l_debt = P_unshocked[1][t]*np.average(n_equity)+Debt_unshocked[1][t]
    l_no_Debt = P_unshocked[0][t]*np.average(n_equity)+Debt_unshocked[0][t]
    L_debt.append(l_debt)
    L_no_Debt.append(l_no_Debt)
    ratio.append(l_debt/l_no_Debt)
Ratio_liabilities_unshocked.append(L_debt);Ratio_liabilities_unshocked.append(L_no_Debt);Ratio_liabilities_unshocked.append(ratio)
###############
# Leverage
d_LEV_unshocked = {'A/E: debt': Leverage_unshocked[1],'A/E: no debt':Leverage_unshocked[0]}
d_LEV_shocked = {'A/E: debt': Leverage_shocked[1],'A/E: no debt':Leverage_shocked[0]}
# Liabilities
d_shocked = {'L-Ratio':Ratio_liabilities_shocked[2],'Growth Ratio':np.append(np.diff(np.log(Ratio_liabilities_shocked[2])),0),'L : no debt':Ratio_liabilities_shocked[1],'L : debt':Ratio_liabilities_shocked[0]}
d_unshocked = {'L-Ratio':Ratio_liabilities_unshocked[2],'Growth Ratio':np.append(np.diff(np.log(Ratio_liabilities_unshocked[2])),0),'L : no debt':Ratio_liabilities_unshocked[1],'L : debt':Ratio_liabilities_unshocked[0]}
# level values (PI no debt, II debt)
d_PI_shocked = {'Value Assets':A_shocked[0],'Price':P_shocked[0],'Roe':R_shocked[0],'Value Debt':Debt_shocked[0],'P-R':Delta_shocked[0]}
d_PI_unshocked = {'Value Assets':A_unshocked[0],'Price':P_unshocked[0],'Roe':R_unshocked[0],'Value Debt':Debt_unshocked[0],'P-R':Delta_unshocked[0]}
d_II_shocked = {'Value Assets':A_shocked[1],'Price':P_shocked[1],'Roe':R_shocked[1],'Value Debt':Debt_shocked[1],'P-R':Delta_shocked[1]}
d_II_unshocked = {'Value Assets':A_unshocked[1],'Price':P_unshocked[1],'Roe':R_unshocked[1],'Value Debt':Debt_unshocked[1],'P-R':Delta_unshocked[1]}
### DATAFRAME STUFF
df_PI_shocked = pd.DataFrame(d_PI_shocked)
df_PI_unshocked = pd.DataFrame(d_PI_unshocked)
df_II_shocked = pd.DataFrame(d_II_shocked)
df_II_unshocked = pd.DataFrame(d_II_unshocked)
df_LEV_shocked = pd.DataFrame(d_LEV_shocked)
df_LEV_unshocked = pd.DataFrame(d_LEV_unshocked)
df_shocked = pd.DataFrame(d_shocked)
df_unshocked = pd.DataFrame(d_unshocked)
### PLOTTING STUFF
df_PI_shocked.plot(title='Equity + No Debt: Imperfect Information - Shock',subplots=True)
df_PI_unshocked.plot(title='Equity + No Debt: Imperfect Information - No Shock',subplots=True)
df_II_shocked.plot(title='Equity + Debt: Imperfect Information - Shock',subplots=True)
df_II_unshocked.plot(title='Equity + Debt: Imperfect Information - No Shock',subplots=True)
df_LEV_shocked.plot(title='Leverage: Imperfect Information - Shock',subplots=False)
df_LEV_unshocked.plot(title='Leverage: Imperfect Information - No Shock',subplots=False)
df_shocked.plot(title='Liabilities: Imperfect Information - Shock')
df_unshocked.plot(title='Liabilities: Imperfect Information - No Shock')
plt.show()
