import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog

def opt_plan_discrete_multiasset(values11,prob11,values12,prob12,values21,prob21,values22,prob22,func,prob1=None,
                                       prob2=None,onedim=True,minimize=True,schmithals=False,correlation_1=False,
                                       corr_1=0.5,correlation_2=False,corr_2=0.5,
                                       basket_prices = None,basket_strikes=None,basket_time = 1,
                                       same_correlation = False, increasing_prices = False,
                                       proportional = False,prop_constant_lower = 0.02,prop_constant_upper = 0.4, t_1 = 1, t_2 =2,prop_range = 0.005,
                                       q_corr_greater_p = False, q_corr_greater_p_const = 0,
                                       copula_indices = [], copula_strikes=[], copula_prices=[],
                                       martingale_condition = True
                                        ):
    n11 = len(values11) # Length of the vector with the values of the first marginal, 1st security
    n21 = len(values21) # Length of the vector with the values of the 2nd marginal, 1st security
    n12 = len(values12) # Length of the vector with the values of the first marginal, 2nd security
    n22 = len(values22) # Length of the vector with the values of the 2nd marginal, 2nd security
    
    # Conversion to np arrays:
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
    
    # Sclaing the probabilities
    prob11 = prob11 / np.sum(prob11)
    prob12 = prob12 / np.sum(prob12)
    prob21 = prob21 / np.sum(prob21)
    prob22 = prob22 / np.sum(prob22)
    

    # Scaling to the same mean
    mean11 = np.sum(values11*prob11)
    mean12 = np.sum(values12*prob12)
    mean21 = np.sum(values21*prob21)
    mean22 = np.sum(values22*prob22)
    values11 = values11 + 0.5*(mean21-mean11)
    values21 = values21 + 0.5*(mean11-mean21)
    values12 = values12 + 0.5*(mean22-mean12)
    values22 = values22 + 0.5*(mean12-mean22)
  

    # Initiate the Gurobi Model
    m = gp.Model("m")
    # No Output
    m.setParam( 'OutputFlag', False )
    # The measure variable
    x = m.addMVar(shape=np.int(n11*n12*n21*n22),lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="x")    
     
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
    
    
    if martingale_condition:    
        if schmithals:
            for i in range(n11):
                a = np.zeros((n11,n12,n21,n22))
                for j in range(n12):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = values21[k]-values11[i]
                #A[i,:] = np.reshape(a, n11*n12*n21*n22)
                m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == 0)            
            for j in range(n12):
                a = np.zeros((n11,n12,n21,n22))
                for i in range(n11):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = values22[l]-values12[j]
                #A[n11*n12+j,:] = np.reshape(a, n11*n12*n21*n22)
                m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == 0)    
        elif schmithals == False:
            for i in range(n11):
                for j in range(n12):
                    a = np.zeros((n11,n12,n21,n22))
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = values21[k]-values11[i]
                    #A[i+(j)*(n11),:] = np.reshape(a, n11*n12*n21*n22)
                    m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == 0)    
            for j in range(n12):
                for i in range(n11):
                    a = np.zeros((n11,n12,n21,n22))
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = values22[l]-values12[j]
                    #A[n11*n12+i+(j)*(n11),:] = np.reshape(a, n11*n12*n21*n22)
                    m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == 0)            
    # marginal Constraints
    for i in range(n11):
        a = np.zeros((n11,n12,n21,n22))
        a[i,:,:,:] = 1
        #A[2*n11*n12+i,] = np.reshape(a, n11*n12*n21*n22)
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == prob11[i]) 
    for i in range(n12):
        a = np.zeros((n11,n12,n21,n22))
        a[:,i,:,:] = 1
        #A[2*n11*n12+n11+i,] = np.reshape(a, n11*n12*n21*n22)
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == prob12[i])     
    for i in range(n21):
        a = np.zeros((n11,n12,n21,n22))
        a[:,:,i,:] = 1
        #A[2*n11*n12+n11+n12+i,] = np.reshape(a, n11*n12*n21*n22)
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == prob21[i]) 
    for i in range(n22):
        a = np.zeros((n11,n12,n21,n22))
        a[:,:,:,i] = 1
        #A[2*n11*n12+n11+n12+n21+i,] = np.reshape(a, n11*n12*n21*n22)
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == prob22[i]) 
    
    # Additional Constraints
    if onedim == False:
        for i in range(n11):
            for j in range(n12):
                a = np.zeros((n11,n12,n21,n22))
                a[i,j,:,:] = 1
                m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == prob1[i+j*n11]) 
                #A[2*n11*n12+n11+n12+n21+n22+i+(j)*n11,:] = np.reshape(a, n11*n12*n21*n22)
        for i in range(n21):
            for j in range(n22):
                a = np.zeros((n11,n12,n21,n22))
                a[:,:,i,j] = 1
                m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == prob2[i+j*n21]) 
                #A[2*n11*n12+n11+n12+n21+n22+n11*n12+i+(j)*n21,:] = np.reshape(a, n11*n12*n21*n22)  
    
    if correlation_1:
        a = np.zeros((n11,n12,n21,n22))
        for i in range(n11):
            for j in range(n12):
                for k in range(n21):
                    for l in range(n22):
                        a[i,j,k,l] = values11[i]*values12[j]
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == corr_1*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)+S_0_1*S_0_2) 
        #A = np.vstack([A, np.reshape(a, n11*n12*n21*n22)])
    if correlation_2:
        a = np.zeros((n11,n12,n21,n22))
        for i in range(n11):
            for j in range(n12):
                for k in range(n21):
                    for l in range(n22):
                        a[i,j,k,l] = values21[k]*values22[l]
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == corr_2*np.sqrt(second_moment_1-S_0_1**2)*np.sqrt(second_moment_2-S_0_2**2)+S_0_1*S_0_2) 
        #A = np.vstack([A, np.reshape(a, n11*n12*n21*n22)])
        
    # Directly incorporate Basket Option Prices
    if basket_prices != None and basket_strikes != None and basket_time == 1:
        for p,s in zip(basket_prices,basket_strikes):
            a = np.zeros((n11,n12,n21,n22))
            for i in range(n11):
                for j in range(n12):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = max(0.5*values11[i]+0.5*values12[j]-s,0)
            m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == p)
            
    if basket_prices != None and basket_strikes != None and basket_time == 2:
        for p,s in zip(basket_prices,basket_strikes):
            a = np.zeros((n11,n12,n21,n22))
            for i in range(n11):
                for j in range(n12):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = max(0.5*values21[k]+0.5*values22[l]-s,0)
            m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == p)
            
    # Assumption on same Correlation
    if same_correlation:
        S_0_1 = np.sum(values11*prob11)
        S_0_2 = np.sum(values12*prob12)
        second_moment_11 = np.sum(values11**2*prob11)
        second_moment_12 = np.sum(values12**2*prob12)
        second_moment_21 = np.sum(values21**2*prob21)
        second_moment_22 = np.sum(values22**2*prob22)
        a = np.zeros((n11,n12,n21,n22))
        for i in range(n11):
            for j in range(n12):
                for k in range(n21):
                    for l in range(n22):
                        a[i,j,k,l] = (values11[i]*values12[j]-S_0_1*S_0_2)/(np.sqrt(second_moment_11-S_0_1**2)*np.sqrt(second_moment_12-S_0_2**2))-(values21[k]*values22[l]-S_0_1*S_0_2)/(np.sqrt(second_moment_21-S_0_1**2)*np.sqrt(second_moment_22-S_0_2**2))
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == 0) 
        
    if increasing_prices:
        S_0_1 = np.sum(values11*prob11)
        S_0_2 = np.sum(values12*prob12)
        S_0 = S_0_1 + S_0_2
        a = np.zeros((n11,n12,n21,n22))
        for K in np.linspace(0,S_0*2,100):
            for i in range(n11):
                for j in range(n12):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = max(values21[k]+values22[l]-K,0)-max(values11[i]+values12[j]-K,0)
            m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x >= 0)     
    
    if proportional:
        S_0_1 = np.sum(values11*prob11)
        S_0_2 = np.sum(values12*prob12)
        S_0 = S_0_1 + S_0_2
        a = np.zeros((n11,n12,n21,n22))
        b = np.zeros((n11,n12,n21,n22))
        for K in np.linspace(S_0*(1-prop_range),S_0*(1+prop_range),100):
            for i in range(n11):
                for j in range(n12):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = max(values21[k]+values22[l]-K,0)
                            b[i,j,k,l] = max(values11[i]+values12[j]-K,0)
            m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x >= prop_constant_lower*np.sqrt(t_2)*S_0)
            m.addConstr(np.reshape(b, n11*n12*n21*n22) @ x >= prop_constant_lower*np.sqrt(t_1)*S_0)
            m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x <= prop_constant_upper*np.sqrt(t_2)*S_0)       
            m.addConstr(np.reshape(b, n11*n12*n21*n22) @ x <= prop_constant_upper*np.sqrt(t_1)*S_0)
            
    if q_corr_greater_p:
        S_0_1 = np.sum(values11*prob11)
        S_0_2 = np.sum(values12*prob12)
        second_moment_11 = np.sum(values11**2*prob11)
        second_moment_12 = np.sum(values12**2*prob12)
        second_moment_21 = np.sum(values21**2*prob21)
        second_moment_22 = np.sum(values22**2*prob22)
        a = np.zeros((n11,n12,n21,n22))
        for i in range(n11):
            for j in range(n12):
                for k in range(n21):
                    for l in range(n22):
                        a[i,j,k,l] = (values11[i]*values12[j]-S_0_1*S_0_2)/(np.sqrt(second_moment_11-S_0_1**2)*np.sqrt(second_moment_12-S_0_2**2))
        m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x >= q_corr_greater_p_const)
    
    counter = 0
    for ind in copula_indices:
        # indicator for the indices
        i11 = (ind[0]==1)*(ind[2]==1)+(ind[1]==1)*(ind[3]==1)
        i12 = (ind[0]==1)*(ind[2]==2)+(ind[1]==1)*(ind[3]==2)
        i21 = (ind[0]==2)*(ind[2]==1)+(ind[1]==2)*(ind[3]==1)
        i22 = (ind[0]==2)*(ind[2]==2)+(ind[1]==2)*(ind[3]==2)   
        a = np.zeros((n11,n12,n21,n22))
        for K in range(len(copula_strikes[counter])):
            for i in range(n11):
                for j in range(n12):
                    for k in range(n21):
                        for l in range(n22):
                            a[i,j,k,l] = (np.max([values11[i]*i11,values12[j]*i12,values21[k]*i21,values22[l]*i22]) <= copula_strikes[counter][K])
            m.addConstr(np.reshape(a, n11*n12*n21*n22) @ x == copula_prices[counter][K])
        counter+=1
        
    costs = np.zeros((n11,n12,n21,n22))
    for i in range(n11):
        for j in range(n12):
            for k in range(n21):
                for l in range(n22):
                    costs[i,j,k,l] = func(values11[i],values12[j],values21[k],values22[l]) 
    costs = np.reshape(costs,n11*n12*n21*n22)
   #print(costs)
    if minimize == True:
        m.setObjective(costs @ x, GRB.MINIMIZE)
    elif minimize == False:
        m.setObjective(costs @ x, GRB.MAXIMIZE)
    m.optimize()
    price = m.objVal
    q = [v.x for v in m.getVars()]
        
    return price, q


def dual_multiasset_2dim(values11,prob11,values12,prob12,values21,prob21,values22,prob22,func,prob1=None,
                                       prob2=None,onedim=True,minimize=True,schmithals=False,correlation_1=False,
                                       corr_1=0.5,correlation_2=False,corr_2=0.5,
                                       basket_prices = None,basket_strikes=None):
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

# N = 11

# # First Security
# p11 = np.repeat(1/N,N)
# v11 = np.linspace(-1,1,N)
# p11[0] = (1/(2*N))
# p11[-1] = (1/(2*N))
# p11 = p11/(np.sum(p11))
# p21 = np.repeat(1/N,N)
# v21 = np.linspace(-3,3,N)
# p21[0] = (1/(2*N))
# p21[-1] = (1/(2*N))
# p21 = p21/(np.sum(p21))
# # Second Security
# p12 = np.repeat(1/N,N)
# v12 = np.linspace(-1,1,N)
# p12[0] = (1/(2*N))
# p12[-1] = (1/(2*N))
# p12 = p12/(np.sum(p12))
# p22 = np.repeat(1/N,N)
# v22 = np.linspace(-2,2,N)
# p22[0] = (1/(2*N))
# p22[-1] = (1/(2*N))
# p22 = p22/(np.sum(p22))

# v11 = margs_00[0]
# p11 = margs_00[1]

# v12 = margs_01[0]
# p12 = margs_01[1]

# v21 = margs_10[0]
# p21 = margs_10[1]

# v22 = margs_11[0]
# p22 = margs_11[1]

# K  = -1
# p = 2
# def payoff1(x_1,x_2,x_3,x_4):
#     return abs(x_3-x_4)**p

# print(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,func=payoff1,onedim=True,minimize=True,schmithals=False)[0])
# print(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,func=payoff1,onedim=True,minimize=False,schmithals=False)[0])