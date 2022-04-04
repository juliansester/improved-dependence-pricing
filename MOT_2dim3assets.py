import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog

def opt_3assets(values11,prob11,values12,prob12,values13,prob13,
                values21,prob21,values22,prob22,values23,prob23,
                func,minimize=True,
                schmithals=False,
                copula_indices = [], copula_strikes=[], copula_prices=[],
                martingale_condition = True,
                only_second_time = False):
    
    n11 = len(values11) # Length of the vector with the values of the first marginal, 1st security
    n21 = len(values21) # Length of the vector with the values of the 2nd marginal, 1st security
    n12 = len(values12) # Length of the vector with the values of the first marginal, 2nd security
    n22 = len(values22) # Length of the vector with the values of the 2nd marginal, 2nd security
    n13 = len(values13) # Length of the vector with the values of the first marginal, 3rd security
    n23 = len(values23) # Length of the vector with the values of the 2nd marginal, 3rd security
    N = n11*n12*n13*n21*n22*n23
   
    # Conversion to np arrays:
    values11 = np.array(values11)
    prob11 = np.array(prob11)
    values12 = np.array(values12)
    prob12 = np.array(prob12)
    values21 = np.array(values21)
    prob21 = np.array(prob21)
    values22 = np.array(values22)
    prob22 = np.array(prob22)
    values23 = np.array(values23)
    prob23 = np.array(prob23)
    values23 = np.array(values23)
    prob23 = np.array(prob23)
    
    # Normalizing the probabilities
    prob11 = prob11 / np.sum(prob11)
    prob12 = prob12 / np.sum(prob12)
    prob13 = prob13 / np.sum(prob13)
    prob21 = prob21 / np.sum(prob21)
    prob22 = prob22 / np.sum(prob22)
    prob23 = prob23 / np.sum(prob23)    

    # Scaling to the same mean
    mean11 = np.sum(values11*prob11)
    mean12 = np.sum(values12*prob12)
    mean13 = np.sum(values13*prob13)
    mean21 = np.sum(values21*prob21)
    mean22 = np.sum(values22*prob22)
    mean23 = np.sum(values23*prob23)

    values11 = values11 + 0.5*(mean21-mean11)
    values21 = values21 + 0.5*(mean11-mean21)

    values12 = values12 + 0.5*(mean22-mean12)
    values22 = values22 + 0.5*(mean12-mean22)

    values13 = values13 + 0.5*(mean23-mean13)
    values23 = values23 + 0.5*(mean13-mean23)

    # Initiate the Gurobi Model
    model = gp.Model("m")
    # No Output
    model.setParam( 'OutputFlag', False )
    # The measure variable
    x = model.addMVar(shape=np.int(N),lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="x")    
     
    if only_second_time == False:
        if martingale_condition:
            if schmithals:
                for i in range(n11):
                    a = np.zeros((n11,n12,n13,n21,n22,n23))
                    for j in range(n12):
                        for m in range(n13):
                            for k in range(n21):
                                for l in range(n22):
                                    for n in range(n23):
                                        a[i,j,m,k,l,n] = values21[k]-values11[i]
                    model.addConstr(np.reshape(a, N) @ x == 0)
                    
                for j in range(n12):
                    a = np.zeros((n11,n12,n13,n21,n22,n23))
                    for i in range(n11):
                        for m in range(n13):
                            for k in range(n21):
                                for l in range(n22):
                                    for n in range(n23):
                                        a[i,j,m,k,l,n] = values22[l]-values12[j]
                    model.addConstr(np.reshape(a, N) @ x == 0)  
                    
                for m in range(n13):
                    a = np.zeros((n11,n12,n13,n21,n22,n23))
                    for i in range(n11):
                        for j in range(n12):
                            for k in range(n21):
                                for l in range(n22):
                                    for n in range(n23):
                                        a[i,j,m,k,l,n] = values23[n]-values13[m]
                    model.addConstr(np.reshape(a, N) @ x == 0)        
                    
            elif schmithals == False:
                for i in range(n11):
                    for j in range(n12):
                        for m in range(n13):
                            a = np.zeros((n11,n12,n13,n21,n22,n23))
                            for k in range(n21):
                                for l in range(n22):
                                    for n in range(n23):
                                        a[i,j,m,k,l,n] = values21[k]-values11[i]
                            model.addConstr(np.reshape(a, N) @ x == 0)
                        
                for j in range(n12):
                    for i in range(n11):
                        for m in range(n13):
                            a = np.zeros((n11,n12,n13,n21,n22,n23))
                            for k in range(n21):
                                for l in range(n22):
                                    for n in range(n23):
                                        a[i,j,m,k,l,n] = values22[l]-values12[j]
                            model.addConstr(np.reshape(a, N) @ x == 0)
                                    
                for j in range(n12):
                    for i in range(n11):
                        for m in range(n13):
                            a = np.zeros((n11,n12,n13,n21,n22,n23))
                            for k in range(n21):
                                for l in range(n22):
                                    for n in range(n23):
                                        a[i,j,m,k,l,n] = values23[n]-values13[m]
                            model.addConstr(np.reshape(a, N) @ x == 0)                            
                            
    # marginal Constraints
    if only_second_time == False:
        for i in range(n11):
            a = np.zeros((n11,n12,n13,n21,n22,n23))
            a[i,:,:,:,:,:] = 1
            model.addConstr(np.reshape(a, N) @ x == prob11[i]) 
        for i in range(n12):
            a = np.zeros((n11,n12,n13,n21,n22,n23))
            a[:,i,:,:,:,:] = 1
            model.addConstr(np.reshape(a, N) @ x == prob12[i])
        for i in range(n13):
            a = np.zeros((n11,n12,n13,n21,n22,n23))
            a[:,:,i,:,:,:] = 1
            model.addConstr(np.reshape(a, N) @ x == prob13[i])        
    for i in range(n21):
        a = np.zeros((n11,n12,n13,n21,n22,n23))
        a[:,:,:,i,:,:] = 1
        model.addConstr(np.reshape(a, N) @ x == prob21[i]) 
    for i in range(n22):
        a = np.zeros((n11,n12,n13,n21,n22,n23))
        a[:,:,:,:,i,:] = 1
        model.addConstr(np.reshape(a, N) @ x == prob22[i]) 
    for i in range(n23):
        a = np.zeros((n11,n12,n13,n21,n22,n23))
        a[:,:,:,:,:,i] = 1
        model.addConstr(np.reshape(a, N) @ x == prob23[i]) 
        
    
    
    counter = 0
    for ind in copula_indices:
        # indicator for the indices
        i11 = (ind[0]==1)*(ind[2]==1)+(ind[1]==1)*(ind[3]==1)
        i12 = (ind[0]==1)*(ind[2]==2)+(ind[1]==1)*(ind[3]==2)
        i13 = (ind[0]==1)*(ind[2]==3)+(ind[1]==1)*(ind[3]==3)
        i21 = (ind[0]==2)*(ind[2]==1)+(ind[1]==2)*(ind[3]==1)
        i22 = (ind[0]==2)*(ind[2]==2)+(ind[1]==2)*(ind[3]==2)
        i23 = (ind[0]==2)*(ind[2]==3)+(ind[1]==2)*(ind[3]==3)
        a = np.zeros((n11,n12,n13,n21,n22,n23))
        for K in range(len(copula_strikes[counter])):
            for i in range(n11):
                for j in range(n12):
                    for m in range(n13):
                        for k in range(n21):
                            for l in range(n22):
                                for n in range(n23):
                                    max_list = [values11[i]*i11,
                                                values12[j]*i12,
                                                values13[m]*i13,
                                                values21[k]*i21,
                                                values22[l]*i22,
                                                values23[n]*i23]
                                    a[i,j,m,k,l,n] = (np.amax(max_list,0) <= copula_strikes[counter][K])
            model.addConstr(np.reshape(a, N) @ x == copula_prices[counter][K])
        counter+=1
        
    costs = np.zeros((n11,n12,n13,n21,n22,n23))
    for i in range(n11):
        for j in range(n12):
            for m in range(n13):
                for k in range(n21):
                    for l in range(n22):
                        for n in range(n23):
                            costs[i,j,m,k,l,n] = func(values11[i],
                                                      values12[j],
                                                      values13[m],
                                                      values21[k],
                                                      values22[l],
                                                      values23[n])
    costs = np.reshape(costs,N)
   #print(costs)
    if minimize == True:
        model.setObjective(costs @ x, GRB.MINIMIZE)
    elif minimize == False:
        model.setObjective(costs @ x, GRB.MAXIMIZE)
    model.optimize()
    price = model.objVal
    q = [v.x for v in model.getVars()]
        
    return price, q

