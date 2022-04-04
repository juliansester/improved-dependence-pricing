
import numpy as np
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB

def dual_multiasset_2dim(values11,prob11,values12,prob12,values21,prob21,values22,prob22,func,prob1=None,
                                       prob2=None,onedim=True,minimize=True,schmithals=False,correlation_1=False,
                                       corr_1=0.5,correlation_2=False,corr_2=0.5):
    n11 = len(values11) # Length of the vector with the values of the first marginal, 1st security
    n21 = len(values21) # Length of the vector with the values of the 2nd marginal, 1st security
    n12 = len(values12) # Length of the vector with the values of the first marginal, 2nd security
    n22 = len(values22) # Length of the vector with the values of the 2nd marginal, 2nd security
    
    # Conversion to np.arrays:
    values11 = np.array(values11)
    prob11 = np.array(prob11)
    values12 = np.array(values12)
    prob12 = np.array(prob12)
    values21 = np.array(values21)
    prob21 = np.array(prob21)
    values22 = np.array(values22)
    prob22 = np.array(prob22)
    prob1 = np.array(prob1)
    prob2 = np.array(prob2)
    costs = np.zeros((n11,n12,n21,n22))
    
    if correlation_1:
        S_0_1 = np.sum(values11*prob11)
        S_0_2 = np.sum(values12*prob12)
        second_moment_1 = np.sum(values11**2*prob11)
        second_moment_2 = np.sum(values12**2*prob12)
        #r = np.concatenate((r,corr_1*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)+S_0_1*S_0_2))
        
    if correlation_2:
        S_0_1 = np.sum(values21*prob21)
        S_0_2 = np.sum(values22*prob22)
        second_moment_1 = np.sum(values21**2*prob21)
        second_moment_2 = np.sum(values22**2*prob22)
        #r = np.concatenate((r,corr_2*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)+S_0_1*S_0_2))
        
    # Defining Cost Function
    for i in range(n11):
        for j in range(n12):
            for k in range(n21):
                for l in range(n22):
                    costs[i,j,k,l] = func(values11[i],values12[j],values21[k],values22[l])  
                    
    # Defining Length of the Variables              
    if schmithals:
        length_of_trading_variables = n11+n12
    elif schmithals == False:    
        length_of_trading_variables = 2*n11*n12
    if correlation_1:
        length_of_trading_variables+=1
    if correlation_2:
        length_of_trading_variables+=1
    if onedim:
        length_of_static_variables = n11+n12+n21+n22
    elif onedim == False:
        length_of_static_variables = n11*n12 + n21*n22       
    
    nr_of_variables = length_of_static_variables + length_of_trading_variables
        
    
    
    # Gurobi Model
    m = gp.Model("m")
    m.setParam( 'OutputFlag', False )
    x = m.addMVar(shape=nr_of_variables,lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")

    
    # Defining the Conditions
    for i in range(n11):
        for j in range(n12):
            for k in range(n21):
                for l in range(n22):
                    if onedim == True:
                        a1 = np.repeat(0,n11)
                        a1[i] = 1
                        a2 = np.repeat(0,n12)
                        a2[j] = 1
                        a3 = np.repeat(0,n21)
                        a3[k] = 1
                        a4 = np.repeat(0,n22)
                        a4[l] = 1
                        if schmithals:
                            lhs =  np.concatenate((a1,a2,a3,a4,a1*(values21[k]-values11[i]),a2*(values22[l]-values12[j])))
                        elif schmithals == False:
                            a5= np.zeros((n11,n12))
                            a5[i,j] = 1
                            a5=np.reshape(a5,n11*n12)
                            lhs =  np.concatenate((a1,a2,a3,a4,a5*(values21[k]-values11[i]),a5*(values22[l]-values12[j])))
                        if correlation_1:
                            add = np.reshape(values11[i]*values12[j]
                                             -corr_1*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)-S_0_1*S_0_2,1)
                            lhs = np.concatenate((lhs,add))
                        if correlation_2:
                            add = np.reshape(values21[k]*values22[l]-corr_2*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)-S_0_1*S_0_2,1)
                            lhs = np.concatenate((lhs,add))
                        if minimize == True:
                            m.addConstr(lhs @ x <= np.array(costs[i,j,k,l]))
                        elif minimize == False:
                            m.addConstr(lhs @ x >= np.array(costs[i,j,k,l]))
                    elif onedim == False:
                        a1 = np.repeat(0,n11)
                        a1[i] = 1
                        a2 = np.repeat(0,n12)
                        a2[j] = 1
                        a3 = np.zeros((n11,n12))
                        a3[i,j] = 1
                        a3 = np.reshape(a3,n11*n12)
                        a4 = np.zeros((n21,n22))
                        a4[k,l] = 1
                        a4 = np.reshape(a4,n21*n22)
                        if schmithals:
                            lhs =  np.concatenate((a3,a4,a1*(values21[k]-values11[i]),a2*(values22[l]-values12[j])))
                        elif schmithals == False:
                            a5= np.zeros((n11,n12))
                            a5[i,j] = 1
                            a5=np.reshape(a5,n11*n12)
                            lhs =  np.concatenate((a3,a4,a5*(values21[k]-values11[i]),a5*(values22[l]-values12[j])))
                        if correlation_1:
                            add = np.reshape(values11[i]*values12[j]
                                             -corr_1*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)-S_0_1*S_0_2,1)
                            lhs = np.concatenate((lhs,add))
                        if correlation_2:
                            add = np.reshape(values21[k]*values22[l]-corr_2*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)-S_0_1*S_0_2,1)
                            lhs = np.concatenate((lhs,add))
                        if minimize == True:
                            m.addConstr(lhs @ x <= np.array(costs[i,j,k,l]))
                        elif minimize == False:
                            m.addConstr(lhs @ x >= np.array(costs[i,j,k,l]))
                            
    # Solve Linear System
    #####################
    if onedim:
        objective = np.concatenate((prob11,prob12,prob21,prob22,np.zeros(length_of_trading_variables)))
    elif onedim == False:
        objective = np.concatenate((prob1,prob2,np.zeros(length_of_trading_variables)))
    if(minimize == True):
        m.setObjective(objective @ x, GRB.MAXIMIZE)
    elif(minimize == False):
        m.setObjective(objective @ x, GRB.MINIMIZE)
    m.optimize()
    return m.objVal, x.X



def dual_2dim_prices(strikes1_asset1,prices1_asset1,strikes2_asset1,prices2_asset1,
                     strikes1_asset2,prices1_asset2,strikes2_asset2,prices2_asset2,
                     func,s0_asset1=100,s0_asset2=100,discretization_points = 10,
                      from_discrete= 0.5, to_discrete = 2, minimize=True):
    
    strikes1_asset1.append(0)
    strikes2_asset1.append(0)
    prices1_asset1.append(s0_asset1)
    prices2_asset1.append(s0_asset1)
    
    strikes1_asset2.append(0)
    strikes2_asset2.append(0)
    prices1_asset2.append(s0_asset2)
    prices2_asset2.append(s0_asset2)
    
    n1_asset1 = len(strikes1_asset1)
    n2_asset1 = len(strikes2_asset1)
    n1_asset2 = len(strikes1_asset2)
    n2_asset2 = len(strikes2_asset2)
    
    nr_of_variables = n1_asset1+n1_asset2+n2_asset1+n2_asset2
    
    # shortcut to make the code shorter:
    dp = discretization_points
    
    # Setting the Grid where we evaluate the function
    values1_asset1 = np.linspace(from_discrete*s0_asset1,to_discrete*s0_asset1,dp)
    values2_asset1 = np.linspace(from_discrete*s0_asset1,to_discrete*s0_asset1,dp)
    values1_asset2 = np.linspace(from_discrete*s0_asset2,to_discrete*s0_asset2,dp)
    values2_asset2 = np.linspace(from_discrete*s0_asset2,to_discrete*s0_asset2,dp)        
    
    
    # Defining the Cost Function
    costs = np.zeros((dp,dp,dp,dp))
    for i in range(dp):
        for j in range(dp):
            for k in range(dp):
                for l in range(dp):
                    costs[i,j,k,l] = func(values1_asset1[i],values2_asset1[j],values1_asset2[k],values2_asset2[l])
    m = gp.Model("m")
    m.setParam( 'OutputFlag', True )
    x = m.addMVar(shape=1+nr_of_variables+2*dp**2,lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
     
    for i in range(dp):
        for j in range(dp):
            for ii in range(dp):
                for jj in range(dp):
                    for k in range(dp):
                        a1 = np.repeat(0,n1_asset1)
                        for l in range(n1_asset1):
                            a1[l] = max(values1_asset1[i]-strikes1_asset1[l],0)
                        a2 = np.repeat(0,n2_asset1)
                        for l in range(n2_asset1):
                            a2[l] = max(values2_asset1[j]-strikes2_asset1[l],0)
                        a3 = np.repeat(0,n1_asset2)
                        for l in range(n1_asset2):
                            a1[l] = max(values1_asset2[ii]-strikes1_asset2[l],0)
                        a4 = np.repeat(0,n2_asset2)
                        for l in range(n2_asset2):
                            a2[l] = max(values2_asset2[jj]-strikes2_asset2[l],0)                        
                        a5 = np.zeros((dp,dp))
                        a5[i,ii] = 1
                        a5 = np.reshape(a5,dp**2)
                        lhs =  np.concatenate(([1],a1,a2,a3,a4,
                                               a5*(values2_asset1[j]-values1_asset1[i]),a5*(values2_asset2[j]-values1_asset2[i])))
                        if minimize == True:
                            m.addConstr(lhs @ x <= np.array(costs[i,j,ii,jj]))
                        elif minimize == False:
                            m.addConstr(lhs @ x >= np.array(costs[i,j,ii,jj]))
        
    #print(np.unique(A, return_counts=True))
    # Solve Linear System
    #####################
    objective = np.concatenate(([1],prices1_asset1,prices2_asset1,prices1_asset2,prices2_asset2,np.zeros(2*dp**2)))
    if minimize == True:
        m.setObjective(objective @ x, GRB.MAXIMIZE)
    elif minimize == False:
        m.setObjective(objective @ x, GRB.MINIMIZE)
    m.optimize()
    return m.objVal, m

