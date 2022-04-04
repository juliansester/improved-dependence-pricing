# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:58:26 2020

@author: Julian
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom
import itertools

def opt_plan_discrete_multiasset_n(values_1,prob_1,values_2,prob_2,func,minimize=True,martingale = True,schmithals=False,
                                       same_correlation = False, increasing_prices = False,
                                       proportional = False, prop_constant_upper = 4, prop_constant_lower = 0,prop_range = 0.005,
                                       q_corr_greater_p = False, q_corr_greater_p_const = 0
                                       ):   
    
    n_1 = len(values_1)
    n_2 = len(values_2)
    n = n_1 + n_2
                    
    # Define necessary Variables: Length of the vectors
    N_1=[len(values_1[i]) for i in range(n_1)]
    N_2=[len(values_2[i]) for i in range(n_2)]
    N = N_1 + N_2

    # Initiate the Gurobi Model
    m = gp.Model("m")
    # No Output
    m.setParam( 'OutputFlag', False )
    # The measure variable
    x = m.addMVar(shape=np.int(np.prod(N_1)*np.prod(N_2)),lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="x")

    indices_in_one_list_1 = []
    for i in range(n_1):
        for j in range(N_1[i]):
                indices_in_one_list_1.append(j)
 
    indices_in_one_list_2 = []
    for i in range(n_2):
        for j in range(N_2[i]):
                indices_in_one_list_2.append(j)    
 
    indices_in_one_list = []
    for i in range(n):
        for j in range(N[i]):
                indices_in_one_list.append(j)
                
    # Dimensions of the Problem as Tuple:
    dimensions = ()
    for i in range(n_1):
        dimensions+=(N_1[i],)
    for i in range(n_2):
        dimensions+=(N_2[i],)
        
        
    #For indexing we need the index range:        
    index_range_1 = ()
    for i in range(n_1):
        index_range_1 +=(slice(0,N_1[i]),)
        
    index_range_2 = ()
    for i in range(n_2):
        index_range_2 +=(slice(0,N_2[i]),)
        
    index_range= ()
    for i in range(n):
        index_range +=(slice(0,N[i]),)
        
    # # Creating a list with all posible tuples of index combinations  
    all_tuples_1=[]
    # iterate over all combinations of length n
    for l in list(itertools.combinations(indices_in_one_list_1,n_1)):
        logic = 1
        # Check whether index i comes from marginal i
        for i in range(n_1):
            logic = logic*(l[i] in range(N_1[i]))
        if logic == 1 and l not in all_tuples_1:
            all_tuples_1.append(l)

    all_tuples_2=[]
    # iterate over all combinations of length n
    for l in list(itertools.combinations(indices_in_one_list_2,n_2)):
        logic = 1
        # Check whether index i comes from marginal i
        for i in range(n_2):
            logic = logic*(l[i] in range(N_2[i]))
        if logic == 1 and l not in all_tuples_2:
            all_tuples_2.append(l)
     
    all_tuples=[]
    for l in list(itertools.combinations(indices_in_one_list,n)):
        logic = 1
        # Check whether index i comes from marginal i
        for i in range(n):
            logic = logic*(l[i] in range(N[i]))
        if logic == 1 and l not in all_tuples:
            all_tuples.append(l)
 

    
    # All tuples of shorter length with same beginning, i.e. cutting the longer tuples  
    tuple_shorter_1=[]    
    for i in range(0,n_1):
        tuple_shorter_1.append([])
        for t in all_tuples_1:
            if t[:i] not in tuple_shorter_1[i]:
                tuple_shorter_1[i].append(t[:i])

    tuple_shorter_2=[]    
    for i in range(0,n_2):
        tuple_shorter_2.append([])
        for t in all_tuples_2:
            if t[:i] not in tuple_shorter_2[i]:
                tuple_shorter_2[i].append(t[:i])            

    tuple_shorter=[]    
    for i in range(0,n):
        tuple_shorter.append([])
        for t in all_tuples:
            if t[:i] not in tuple_shorter[i]:
                tuple_shorter[i].append(t[:i])
                
    # Marginal Conditions  
    ####################
    for i in range(n_1):
        for j in range(N_1[i]):
            ind=()
            a=np.zeros(dimensions)
            for k in range(n_1):
                if k != i :
                    ind += (index_range_1[k],)
                elif k == i :
                    ind += (j,)
            for k in range(n_2):
                ind += (index_range_2[k],)
            a[ind] = 1
            m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == np.array(prob_1[i][j]))

    for i in range(n_2):
        for j in range(N_2[i]):
            ind=()
            a=np.zeros(dimensions)
            for k in range(n_1):
                ind += (index_range_1[k],)          
            for k in range(n_2):
                if k != i :
                    ind += (index_range_2[k],)
                elif k == i :
                    ind += (j,)
            a[ind] = 1
            m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == np.array(prob_2[i][j]))
            
    # Martingale Conditions       
    #######################
    # Condition: E_Q[S_i|S_{i-1},...,S_1]=S_{i-1} f.a. i=1,...,n
    if (schmithals):    
        for i in range(1,n_1): # Loop over time steps
          a = np.zeros(dimensions) # will be fed with data for the measure
          for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
              for tup2 in all_tuples:
                  if tup2[:i] == tup1:
                      a[tup2] = values_1[i][tup2[i]]-values_1[i-1][tup2[i-1]]                      
              # append this condition
              m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == 0)
              
        for i in range(1,n_2): # Loop over time steps
          a = np.zeros(dimensions) # will be fed with data for the measure
          for tup1 in tuple_shorter_2[i]: # Iteration over all possible values
              for tup2 in all_tuples:
                  if tup2[n_1:(n_1+i)] == tup1:
                      a[tup2] = values_2[i][tup2[n_1+i]]-values_2[i-1][tup2[n_1+i-1]]                      
              # append this condition
              m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == 0)
              
    elif (schmithals == False):  
        for i in range(1,n_1): # Loop over time steps
          a = np.zeros(dimensions) # will be fed with data for the measure
          for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
              for tup2 in tuple_shorter_2[i]:
                  for tup3 in all_tuples:
                      if tup3[:i] == tup1 and tup3[n_1:(n_1+i)] == tup2:
                          a[tup3] = values_1[i][tup3[i]]-values_1[i-1][tup3[i-1]]                      
              # append this condition
                  m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == 0)
              
        for i in range(1,n_2): # Loop over time steps
          a = np.zeros(dimensions) # will be fed with data for the measure
          for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
              for tup2 in tuple_shorter_2[i]:
                  for tup3 in all_tuples:
                      if tup3[:i] == tup1 and tup3[n_1:(n_1+i)] == tup2:
                          a[tup3] = values_2[i][tup3[n_1+i]]-values_2[i-1][tup3[n_1+i-1]]                      
              # append this condition
                  m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == 0)
                 
                    
    ######################## ADDITIONAL CONSTRAINTS #############
    # Assumption on same Correlation
    if same_correlation:
        S_0_1 = np.sum(np.array(values_1[0])*np.array(prob_1[0]))
        S_0_2 = np.sum(np.array(values_2[0])*np.array(prob_2[0]))
        
        second_moment_1 = [np.sum(np.array(values_1[i])**2*np.array(prob_1[i])) for i in range(n_1)]
        second_moment_2 = [np.sum(np.array(values_2[i])**2*np.array(prob_2[i])) for i in range(n_2)]
        
        
        # second_moment_11 = np.sum(values11**2*prob11)
        # second_moment_12 = np.sum(values12**2*prob12)
        # second_moment_21 = np.sum(values21**2*prob21)
        # second_moment_22 = np.sum(values22**2*prob22)
        
        # Loop over time steps
        for i in range(1,n_1):
            a = np.zeros(dimensions)
            for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
                for tup2 in tuple_shorter_2[i]:
                    for tup3 in all_tuples:
                        if tup3[:i] == tup1 and tup3[n_1:(n_1+i)] == tup2:
                           a[tup3] = (values_1[i][tup3[i]]*values_2[i][tup3[n_1+i]]-S_0_1*S_0_2)/(np.sqrt(second_moment_1[i]-S_0_1**2)*np.sqrt(second_moment_2[i]-S_0_2**2))-(values_1[i-1][tup3[i-1]]*values_2[i-1][tup3[n_1+i-1]]-S_0_1*S_0_2)/(np.sqrt(second_moment_1[i-1]-S_0_1**2)*np.sqrt(second_moment_2[i-1]-S_0_2**2))
            m.addConstr(np.reshape(a, np.prod(dimensions)) @ x == 0)       
            
        # for i in range(n11):
        #     for j in range(n12):
        #         for k in range(n21):
        #             for l in range(n22):
        #                 a[i,j,k,l] = (values11[i]*values12[j]-S_0_1*S_0_2)/(np.sqrt(second_moment_11-S_0_1**2)*np.sqrt(second_moment_12-S_0_2**2))-(values21[k]*values22[l]-S_0_1*S_0_2)/(np.sqrt(second_moment_21-S_0_1**2)*np.sqrt(second_moment_22-S_0_2**2))

        
    if increasing_prices:
        S_0_1 = np.sum(np.array(values_1[0])*np.array(prob_1[0]))
        S_0_2 = np.sum(np.array(values_2[0])*np.array(prob_2[0]))
        S_0 = S_0_1 + S_0_2
        for i in range(1,n_1):
            for K in np.linspace(0,S_0*2,100):
                a = np.zeros(dimensions)
                for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
                    for tup2 in tuple_shorter_2[i]:
                        for tup3 in all_tuples:
                            if tup3[:i] == tup1 and tup3[n_1:(n_1+i)] == tup2:           
                                a[tup3] = max(values_1[i][tup3[i]]+values_2[i][tup3[n_1+i]]-K,0)-max(values_1[i-1][tup3[i-1]]+values_2[i-1][tup3[n_1+i-1]]-K,0)
                m.addConstr(np.reshape(a, np.prod(dimensions)) @ x >= 0)
        
    if proportional:
        S_0_1 = np.sum(np.array(values_1[0])*np.array(prob_1[0]))
        S_0_2 = np.sum(np.array(values_2[0])*np.array(prob_2[0]))
        S_0 = S_0_1 + S_0_2
        for i in range(n_1):
            for K in np.linspace(S_0*(1-prop_range),S_0*(1+prop_range),100):
                a = np.zeros(dimensions)
                b = np.zeros(dimensions)
                for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
                    for tup2 in tuple_shorter_2[i]:
                        for tup3 in all_tuples:
                            if tup3[:i] == tup1 and tup3[n_1:(n_1+i)] == tup2:           
                                a[tup3] = max(values_1[i][tup3[i]]+values_2[i][tup3[n_1+i]]-K,0)
                                b[tup3] = max(K-values_1[i][tup3[i]]-values_2[i][tup3[n_1+i]],0)
                m.addConstr(np.reshape(a, np.prod(dimensions)) @ x >= prop_constant_lower*np.sqrt((i+1)/252)*S_0)
                m.addConstr(np.reshape(a, np.prod(dimensions)) @ x <= prop_constant_upper*np.sqrt((i+1)/252)*S_0)
                       
    if q_corr_greater_p:
        S_0_1 = np.sum(np.array(values_1[0])*np.array(prob_1[0]))
        S_0_2 = np.sum(np.array(values_2[0])*np.array(prob_2[0]))        
        second_moment_1 = [np.sum(np.array(values_1[i])**2*np.array(prob_1[i])) for i in range(n_1)]
        second_moment_2 = [np.sum(np.array(values_2[i])**2*np.array(prob_2[i])) for i in range(n_2)]        
        
        for i in range(n_1):
            a = np.zeros(dimensions)
            for tup1 in tuple_shorter_1[i]: # Iteration over all possible values
                for tup2 in tuple_shorter_2[i]:
                    for tup3 in all_tuples:
                        if tup3[:i] == tup1 and tup3[n_1:(n_1+i)] == tup2:
                           a[tup3] = (values_1[i][tup3[i]]*values_2[i][tup3[n_1+i]]-S_0_1*S_0_2)/(np.sqrt(second_moment_1[i]-S_0_1**2)*np.sqrt(second_moment_2[i]-S_0_2**2))
            m.addConstr(np.reshape(a, np.prod(dimensions)) @ x >= q_corr_greater_p_const)   
              
    def ind_to_value(index_tuple):
        v=()
        for i in range(n_1):
            v+=(values_1[i][index_tuple[i]],)
        for i in range(n_2):
            v+=(values_2[i][index_tuple[n_1+i]],)
        return v
     
    costs = np.zeros(dimensions)
    
    
    for tuple in all_tuples:
            costs[tuple] = func(*ind_to_value(tuple))
    costs = np.reshape(costs,np.prod(dimensions))

    if minimize == True:
        m.setObjective(costs @ x, GRB.MINIMIZE)
    elif minimize == False:
        m.setObjective(costs @ x, GRB.MAXIMIZE)
    m.optimize()
    price = m.objVal    
    
    return price 



# # TESTING

# # ### UNIFORM MODEL ########
# v_1 = [np.linspace(100-i,100+i,i+1) for i in range(1,3)]
# p_1 = [np.repeat(1/(i+1),i+1) for i in range(1,3)]


# #### BINOMIAL MODEL #########

# prob_up = 0.5
# prob_down = 1-prob_up

# # Defining the Marginals
# # Correcting by adding the proper probabilities
# v_2 = []
# p_2 = []
# for i in range(1,3):
#     v_2.append(np.linspace(100-i,100+i,i+1))
#     probs = []
#     for j in range(i+1):
#         probs.append(binom.pmf(j, i, prob_up))
#     p_2.append(probs)
    
# def payoff1(a,b,c,d):
#     return max((1/4)*(a+b+c+d)-10,0)
    
# print(opt_plan_discrete_multiasset(v_1,p_1,v_2,p_2,func = payoff1,minimize=True))
# print(opt_plan_discrete_multiasset(v_1,p_1,v_2,p_2,func = payoff1,minimize=False))

# # First Security
# p11 = np.repeat(1/3,3)
# v11 = [8,10,12]
# p21 = np.repeat(1/4,4)
# v21 = [7,9,11,13]
# # Second Security
# p12 = np.repeat(1/3,3)
# v12 = [8,10,12]
# p22 = np.repeat(1/5,5)
# v22 = [4,7,10,13,16]

# v_1 = [v11,v21]
# p_1 = [p11,p21]
# v_2 = [v12,v22]
# p_2 = [p12,p22]
# def payoff1(a,b,c,d):
#     return max((1/4)*(a+b+c+d)-10,0)
# def payoff2(a,b,c,d):
#     return (c>a)*(d>b)
# def payoff3(a,b,c,d):
#     return (1/4)*max(c-a,0)*max(d-b,0)
# def payoff4(a,b,c,d):
#     return ((c-a)/a)**2*((d-b)/b)**2
# def payoff5(a,b,c,d):
#     return max(((c-a)/a)*((d-b)/b),0)

# print(opt_plan_discrete_multiasset(v_1,p_1,v_2,p_2,func = payoff1,minimize=True,
#                                    schmithals = True,
#                                    same_correlation=False,increasing_prices = True))
# print(opt_plan_discrete_multiasset(v_1,p_1,v_2,p_2,func = payoff1,minimize=False,
#                                    schmithals = True,
#                                    same_correlation=False,increasing_prices = True))

