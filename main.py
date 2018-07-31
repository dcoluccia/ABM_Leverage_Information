import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
from scipy.stats import norm
import math

class Equity_Debt:
    """Equity owners trade in equity issued by a leveraged firm."""
    def __init__(self,n_investors,w,n_corporates,n_equity,initial_value,time_to_maturity,discount_rate,face_value,beta,delta,mi,sigma,gamma,eta,noise,rho,iterations):
        self.n_investors = n_investors #number of investors
        self.w = w #original aggregate endowment
        self.n_corporates = n_corporates #number of corporates
        self.n_equity = n_equity #nominal amounts of equity
        self.time_to_maturity = time_to_maturity #time to maturity of corporate debts
        self.discount_rate = discount_rate #discount rate in merton model
        self.face_value = face_value #face value of corporate debts
        self.beta = beta #portfolio responsiveness
        self.delta = delta #list of expectation formation horizon
        self.mi = mi #list of drifs of GBMs
        self.sigma = sigma #list of variances of GBMs
        self.gamma = gamma #weight of value investors
        self.eta = eta #amount of weight of the signl in portfolio weights
        self.noise = noise #variance of signal noise
        self.rho = rho #inertial noise in the signal
        self.iterations = iterations #number of iterations of the model
        ### EQUITY MARKET STORED VARIABLES
        self.e = {} #individual endowments
        self.old_e = {} #one period lagged individual endowments
        self.equity_holdings = {} #individual equity holdings
        self.expected_e = {} #dictionary containing expected Y_t, initialized at every time step
        self.new_weights = [] #list of optimally chosen weights
        self.old_weights = [] #list of actual portfolio weights
        self.noise_matrix = [] #matrix of signals' weights
        ### DIVIDENDS COMPUTATION
        self.initial_value = initial_value #first vlaue of assets
        self.r = [] #corporate equity returns
        self.P = [] #index market price of a sample corporate's equity
        self.true_returns = [] #true returns equity awards
        self.old_true_returns = [] #past true returns on equity
        self.x = [] #price differences used to estimate drift
        self.non_shocked_assets = []
        ### DESCRIPTIVE
        self.debt_all_shocked = []
        self.debt_one_shocked = []
        self.aggregate_endowment = []

    ######## SHOCK RELATED METHODS #########
    def temporary_shock(self):
        shocked_bank = [i for i in range(self.n_corporates)]
        for t in range(self.iterations):
            for i in shocked_bank:
                if t == 25:
                    self.assets[t][i] -= 0.015
                else:
                    pass

    def onecorporate_temporary_shock(self):
        shocked_bank = [range(3)]
        for t in range(self.iterations):
            for i in shocked_bank:
                if t == 25:
                    self.assets[t][i] += 0.02 #= 0.1#
                else:
                    pass

    def permanent_shock(self,time):
        shocked_bank = [i for i in range(self.n_corporates)]
        assets = copy.deepcopy(self.assets)
        for t in range(self.iterations):
            for i in shocked_bank:
                if t >= 25:
                    self.assets[t][i] = assets[t][i] - 0.015
                else:
                    pass

    def adjust_gamma(self,t):
        if t == 1:
            self.gamma -= 0.5

    def adjust_face_values(self,t):
        if t == 1:
            self.face_value = [self.initial_value[j]*0.8 for j in range(self.n_corporates)] #*0.8

    def adjust(self,t):
        if t == 1:
            self.assets = self.non_shocked_assets
            self.P = [[0 for j in range(self.n_corporates)] for t in range(2)]
            for j in range(self.n_corporates):
                for t in range(2):
                    if t == 0:
                        self.P[t][j] = 1
                    else:
                        self.P[t][j] = 1
            self.x = np.array([[0.01 for j in range(self.n_corporates)] for t in range(2)])
            households = [i for i in range(self.n_investors)] #needed because the enumerate function takes list type not range
            weights = [[1/self.n_corporates for i in range(self.n_investors)] for j in range(self.n_corporates)]
            self.old_weights = np.transpose(weights)
            for i in enumerate(households):
                self.e[(i,i)] = self.w / self.n_investors
                self.equity_holdings[(i,i)] = [self.n_equity[j]/self.n_investors for j in range(self.n_corporates)]
            r = [1 for i in range(self.n_corporates)]
            for i in range(self.n_investors):
                self.old_e[(i,i),(i,i)] = np.dot(np.transpose(self.equity_holdings[(i,i),(i,i)]),r)
            self.true_returns = [1 for j in range(self.n_corporates)]

    ######## POPULATION RELATED METHODS #######
    def compute(self,j,p):
        dt = 0.1 # time step of the process
        Mu = self.mi[j] # drift
        Sigma = self.sigma[j] # variance
        r = np.zeros(self.iterations)
        r[0] = p #initial value of assets
        for k in range(1,self.iterations):
            r[k]=r[k-1]*np.exp((Mu-(Sigma*Sigma)/2)*dt+Sigma*np.sqrt(dt)*random.normalvariate(0,1))
        return r #r is the return vector with len(r)=iterations

    def populate_returns(self):
        d = [self.compute(j,self.initial_value[j]) for j in range(self.n_corporates)];self.assets = np.array(np.transpose(d));self.non_shocked_assets = copy.deepcopy(self.assets)
        signal_noise = [[np.random.normal(0,self.noise) for i in range(self.n_investors)] for j in range(self.iterations)]; self.noise_matrix = np.array(signal_noise)
        self.P = [[0 for j in range(self.n_corporates)] for t in range(2)]
        for j in range(self.n_corporates):
            for t in range(2):
                if t == 0:
                    self.P[t][j] = 1
                else:
                    self.P[t][j] = 1
        self.x = np.array([[0.01 for j in range(self.n_corporates)] for t in range(2)])

    def populate_endowments_holdings(self):
        self.true_returns = [0.01 for j in range(self.n_corporates)]
        households = [i for i in range(self.n_investors)] #needed because the enumerate function takes list type not range
        weights = [[1/self.n_corporates for i in range(self.n_investors)] for i in range(self.n_corporates)]
        self.old_weights = np.transpose(weights)
        for i in enumerate(households):
            self.e[(i,i)] = self.w / self.n_investors
            self.equity_holdings[(i,i)] = [self.n_equity[j]/self.n_investors for j in range(self.n_corporates)]
        for i in range(self.n_investors):
            r = [1 for i in range(self.n_corporates)]
            self.old_e[(i,i),(i,i)] = np.dot(np.transpose(self.equity_holdings[(i,i),(i,i)]),r)

    ######## WEIGHTS RELATED METHODS #######
    def compute_weights(self,i,t):
        mean_vector = []
        time_periods = [i in range(t)]
        for j in range(self.n_corporates):
            m = 0
            for time in time_periods:
                m += self.delta[i]*(1-self.delta[i])**time*self.x[t-time][j]
            mean_vector.append(m)
        sharpe_ratio = []
        for j in range(self.n_corporates):
            f = self.mi[j] + self.noise_matrix[t][j]
            num = self.gamma*f + (1-self.gamma)*mean_vector[j]
            sharpe_ratio.append(num)
        sum_weights = 0
        for j in range(self.n_corporates):
            sum_weights += np.exp(self.beta*sharpe_ratio[j])
        weights = []
        for j in range(self.n_corporates):
            weights.append(np.exp(self.beta*sharpe_ratio[j])/sum_weights)
        return weights

    def compute_weight_matrix(self,t):
        weights_matrix = []
        for i in range(self.n_investors):
            weights_matrix.append(self.compute_weights(i,t))
        self.new_weights = np.array(weights_matrix)

    ######## DIVIDENDS RELATED METHODS #######
    def Expected_assets(self,i,j,t):
        dt = 0.1
        time_periods = [i for i in range(t)]
        households = [i for i in range(self.n_investors)]
        mean_vector = []
        for j in range(self.n_corporates):
            m = 0
            for time in time_periods:
                m += self.delta[i]*((1-self.delta[i])**(time))*self.x[t-time][j]
            mean_vector.append(m)
        Expected_assets = []
        for j in range(self.n_corporates):
            GBM = self.initial_value[j]*np.exp(mean_vector[j]*dt*t)
            Expected_assets.append(GBM)
        return Expected_assets

    def Merton_Debt(self,assets,sigma,j,t): #Merton-implied value of bank j debt at time t given its (expected) assets
        dt = 0.1
        if self.face_value[j] != 0:
            d_1_num = np.log(np.divide(assets,self.face_value[j])+(self.discount_rate+0.5*(sigma**2))*(self.time_to_maturity[j]-t*dt))
            d_1_denom = sigma*np.sqrt(self.time_to_maturity[j] - t*dt)
            d_1 = np.divide(d_1_num,d_1_denom)
            d_2 = d_1 - sigma*np.sqrt(self.time_to_maturity[j]-t*dt)
            D = assets*norm.cdf(-1*d_1)+self.face_value[j]*np.exp(-self.discount_rate*(self.time_to_maturity[j]-t*dt)*norm.cdf(d_2))
        else:
            D = 0
        return D

    def expected_dividends(self,i,t):
        time_periods = [i for i in range(t)]
        sigma_vector = []
        for j in range(self.n_corporates):
            s = 0
            m = 0
            for time in time_periods:
                s += np.sqrt(self.delta[i]*(1-self.delta[i])**time*(self.x[t-time][j]-m)**2)
                m += self.delta[i]*(1-self.delta[i])**time*self.x[t-time][j]
            sigma_vector.append(s)
        assets = []
        for j in range(self.n_corporates):
            assets = self.Expected_assets(i,j,t)
        debts = []
        for j in range(self.n_corporates):
            debt = self.Merton_Debt(assets[j],sigma_vector[j],j,t)
            debts.append(debt)
        expected_dividends = []
        for j in range(self.n_corporates):
            # true
            f = np.divide(self.assets[t][j]-self.Merton_Debt(self.assets[t][j],self.sigma[j],j,t),self.n_equity[j]) + self.noise_matrix[t][j]
            # lagged
            s = self.true_returns[j]
            # estimated
            a = np.divide(assets[j]-debts[j],self.n_equity[j])
            div = self.gamma*f + (1-self.gamma)*(self.eta[i]*s + (1-self.eta[i])*a)
            expected_dividends.append(div)
        return expected_dividends

    ######## MARKET RELATED METHODS #######
    def market(self,t):
        households = [i for i in range(self.n_investors)]
        for i in enumerate(households):
            self.expected_e[(i,i)] = 0
        for i in range(self.n_investors):
            expected_div = self.expected_dividends(i,t)
            self.expected_e[(i,i),(i,i)] = np.dot(expected_div,self.equity_holdings[(i,i),(i,i)])
        new_P = []
        for j in range(self.n_corporates):
            expected_demand_t = 0
            demand_t_1 = 0
            for i in range(self.n_investors):
                expected_demand_t += self.new_weights[i][j]*self.expected_e[(i,i),(i,i)]
                demand_t_1 += self.equity_holdings[(i,i),(i,i)][j]*self.P[t][j]
            new_P.append(self.P[t][j]*(expected_demand_t/demand_t_1))
        self.P.append(new_P)

    def update_endowments(self,t):
        self.old_true_returns = copy.deepcopy(self.true_returns)
        self.true_returns = []
        for j in range(self.n_corporates):
            self.true_returns.append(np.divide(self.assets[t][j]-self.Merton_Debt(self.assets[t][j],self.sigma[j],j,t),self.n_equity[j]))
        for i in range(self.n_investors):
            self.e[(i,i),(i,i)] = np.dot(np.transpose(self.equity_holdings[(i,i),(i,i)]),self.true_returns)
        Phi_vector = []
        for j in range(self.n_corporates):
            Sum_equity_present = 0
            Sum_equity_past = 0
            for i in range(self.n_investors):
                Sum_equity_present += self.e[(i,i),(i,i)]*self.new_weights[i][j]/self.P[t+1][j]
                Sum_equity_past += self.equity_holdings[(i,i),(i,i)][j]
            Phi_vector.append(Sum_equity_past/Sum_equity_present)
        for i in range(self.n_investors):
            for j in range(self.n_corporates):
                new_equity = self.new_weights[i][j]*self.e[(i,i),(i,i)]/self.P[t+1][j]
                self.equity_holdings[(i,i),(i,i)][j] = Phi_vector[j]*new_equity
        for i in range(self.n_investors):
            total_equity = 0
            for j in range(self.n_corporates):
                total_equity += self.equity_holdings[(i,i),(i,i)][j]
            for j in range(self.n_corporates):
                self.old_weights[i][j] = self.equity_holdings[(i,i),(i,i)][j] / total_equity
        for i in range(self.n_investors):
            self.old_e[(i,i),(i,i)] = self.e[(i,i),(i,i)]
        new_x = []
        for j in range(self.n_corporates):
            #new_x.append(np.log(np.divide(self.P[t+1][j],self.P[t][j])))
            new_x.append(self.true_returns[j]-self.old_true_returns[j])
            #new_x.append(np.divide(self.true_returns[j] - self.old_true_returns[j],self.P[t+1][j] - self.P[t][j]))
        new_x = np.array(new_x);self.x = np.vstack((self.x,new_x))
        debt = []
        for j in range(self.n_corporates):
            debt.append(self.Merton_Debt(self.assets[t][j],self.sigma[j],j,t))
        self.debt_all_shocked = np.average(debt)
        self.debt_one_shocked = np.array(debt)
        self.aggregate_endowment = 0
        for i in range(self.n_investors):
            self.aggregate_endowment += self.e[(i,i),(i,i)]
