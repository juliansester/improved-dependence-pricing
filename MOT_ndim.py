"""
Martingale Transport, n Marginals 

@author: Julian Sester


Give Marginals in the form 
List of Values of Marginal1,  List of Probabilites of Marginal1, List of Values of Marginal2, List of Probabilites of Marginal2, ....

"""
#from gurobipy import *

def mot(*args,func,minimize=True,martingale = True,variance_info =False, variance_level = 1,variance_times = (1,2), method ="gurobi",iterations = 10e20): 
    # Importing necessary packages
    import numpy as np
    import itertools
    from scipy.optimize import linprog
    
    
    # Grab the Number of Marginals
    n= int(len(args)/2)
    
    # Grab the values and the probabilities
    values=[]
    prob=[]

    for i in range(n):
            values.append(args[2*i])
            prob.append(args[2*i+1])    
                   
    # Define necessary Variables: Length of the vectors
    N=[]
    for i in range(n):
        N.append(len(values[i]))         
   
    # Create one List containing all indices: [i_1,....,i_n,j_1,...j_n,k_1,...k_n,...]
    # where i_1,...i_n are the indices from the first marginal, j_1,...j_n the indices from 
    # the second and so on.
    indices_in_one_list = []
    for i in range(n):
        for j in range(N[i]):
                indices_in_one_list.append(j)
 
    
    # Dimensions of the Problem as Tuple:
    dimensions = ()
    for i in range(n):
        dimensions+=(N[i],)
        
    #For indexing we need the index range:        
    index_range = ()
    for i in range(n):
        index_range +=(slice(0,N[i]),)
        
    # Creating a list with all posible tuples of index combinations  
    all_tuples=[]
    # iterate over all combinations of length n
    for l in list(itertools.combinations(indices_in_one_list,n)):
        logic = 1
        # Check whether index i comes from marginal i
        for i in range(n):
            logic = logic*(l[i] in range(N[i]))
        if logic == 1 and l not in all_tuples:
            all_tuples.append(l)
    
    # Free Memory
    indices_in_one_list = 0
    # All tuples of shorter length with same beginning, i.e. cutting the longer tuples  
    tuple_shorter=[]    
    for i in range(0,n):
        tuple_shorter.append([])
        for t in all_tuples:
            if t[:i] not in tuple_shorter[i]:
                tuple_shorter[i].append(t[:i])           
    if method != "gurobi":
        # Define R.H.S vector
        ####################
        r=[]
        for i in range(n):
            for j in range(N[i]):
                r.append(prob[i][j]) # Adding the rhs values for the marginal conditions
        
        # L.H.S. Vector / Setting the size
        ##################################
        
        A = np.zeros((np.sum(dimensions),np.prod(dimensions))) # Adjusting Size only for Marginal conditions
    
        # Marginal Conditions  
        ####################
        row=0 # indicates the current row of the Matrix A
        for i in range(n):
            for j in range(N[i]):
                ind=()
                a=np.zeros(dimensions)
                for k in range(n):
                    if k != i :
                        ind += (index_range[k],)
                    elif k == i :
                        ind += (j,)
                a[ind] = 1
                A[row,:]=np.reshape(a,np.prod(dimensions))
                row+=1
                
        # Martingale Conditions       
        #######################
        # Condition: E_Q[S_i|S_{i-1},...,S_1]=S_{i-1} f.a. i=1,...,n
        if martingale:
            for i in range(1,n): # Loop over time steps
                a = np.zeros(dimensions) # will be fed with data for the measure
                for tup1 in tuple_shorter[i]: # Iteration over all possible values
                    for tup2 in all_tuples:
                        if tup2[:i] == tup1:
                            a[tup2] = values[i][tup2[i]]-values[i-1][tup2[i-1]]                      
                    # append this condition to A
                    A = np.vstack([A, np.reshape(a,np.prod(dimensions))])
                    # append a 0 to the right hand side
                    r = np.hstack([r,0])
                    row+=1
    elif method == "gurobi":
        # Build the Gurobi Model
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model("m")
        m.setParam( 'OutputFlag', False )
        m.setParam('Method', 1)
        m.setParam('IterationLimit',iterations)
        x = m.addMVar(shape=np.int(np.prod(dimensions)),lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="x") 
        # Marginal Conditions  
        ####################
        row=0 # indicates the current row of the Matrix A
        for i in range(n):
            for j in range(N[i]):
                ind=()
                a=np.zeros(dimensions)
                for k in range(n):
                    if k != i :
                        ind += (index_range[k],)
                    elif k == i :
                        ind += (j,)
                a[ind] = 1
                m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == np.array(prob[i][j]))                 
        # Martingale Conditions       
        #######################
        # Condition: E_Q[S_i|S_{i-1},...,S_1]=S_{i-1} f.a. i=1,...,n
        if martingale: 
            for i in range(1,n): # Loop over time steps
              a = np.zeros(dimensions) # will be fed with data for the measure
              for tup1 in tuple_shorter[i]: # Iteration over all possible values
                  for tup2 in all_tuples:
                      if tup2[:i] == tup1:
                          a[tup2] = values[i][tup2[i]]-values[i-1][tup2[i-1]]                      
                  # append this condition
                  m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == 0) 
 
        # Variance Information
        ###########################
        # Condition: E_Q[(S_n/S_(n-1))^2]=sigma^2+1
        if variance_info:
            time_1 = variance_times[0]
            time_2 = variance_times[1]
            a = np.zeros(dimensions) # will be fed with data for the measure
            for tup in all_tuples:
                a[tup] = (values[time_2-1][tup[time_2-1]]/values[time_1-1][tup[time_1-1]])**2
                # append this condition
            m.addConstr(np.reshape(a,np.prod(dimensions)) @ x == variance_level+1)
                
    # Define Payoff/Cost Array
    #########################
            
    # Function that returns correpsonding values to some given tuple,
    # e.g. (3,5,1,2) - > (values[1][3],values[2][5],values[3][1],values[4][2])
    def ind_to_value(index_tuple):
        v=()
        for i in range(n):
            v+=(values[i][index_tuple[i]],)
        return v
     
    costs = np.zeros(dimensions)    
    for tuple in all_tuples:
            costs[tuple] = func(*ind_to_value(tuple))
    costs = np.reshape(costs,np.prod(dimensions))           
  
   
    # Solve Linear System
    #####################
    if method != "gurobi":
        if(minimize == True):
            res = linprog(costs, A_eq=A, b_eq=r, bounds=(0,1),  options={"disp": False}, method = "interior-point")
        elif(minimize == False):
            res = linprog(-costs, A_eq=A, b_eq=r, bounds=(0,1),  options={"disp": False}, method = "interior-point")
        if res.success:
            # print out optimal q and optimal price
            q = res["x"]
            if(minimize == True):
                price = res["fun"]
            else:
                price = -res["fun"]            
            return price, q
        else:
            print("linprog failed:", res.message)
    elif method == "gurobi":
        if minimize == True:
            m.setObjective(costs @ x, GRB.MINIMIZE)
        elif minimize == False:
            m.setObjective(costs @ x, GRB.MAXIMIZE)
        m.optimize()
        price = m.objVal
        q = m
        return price, q
""" 
v1 = [5,6,7,8,9,10,11,12,13,14,15]
p1 = [1/len(v1)]*len(v1)
v2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
p2 = [1/len(v2)]*len(v2)
def payoff(x,y):
    return max(2*y-x,0)
print("Example 2")
print("Primal")
import matplotlib.pyplot as plt
from matplotlib import rc

def var_function(x,y):
    return ((y/x)**2-1)
min_var = mot(v1,p1,v2,p2,func=var_function,minimize=True)[0]
max_var = mot(v1,p1,v2,p2,func=var_function,minimize=False)[0]
variance = np.linspace(min_var,max_var,50)

min_1 = []
max_1 = []
for i in range(50):
    min_1.append(mot(v1,p1,v2,p2,func=payoff,minimize=True,variance_info = True, variance_level = variance[i])[0])
    max_1.append(mot(v1,p1,v2,p2,func=payoff,minimize=False,variance_info = True, variance_level = variance[i])[0])
    
plt.plot(variance,min_1, color = "red", label = "Lower Bound")
plt.plot(variance,max_1, color = "blue", label = "Upper Bound")
plt.ylabel('Price Bounds')
plt.xlabel('Variance')
plt.legend(loc="best", frameon=True)
plt.show()
print(mot(v1,p1,v2,p2,func=payoff,minimize=True,variance_info = True, variance_level = 0.15)[0])
print(mot(v1,p1,v2,p2,func=payoff,minimize=False,variance_info =True, variance_level = 0.15)[0])
"""
"""
v = []
p = []

def payoff_2(x8,x9):
    return np.maximum(x9-x8,0)
def payoff_3(x7,x8,x9):
    return np.maximum(x9-x8,0)
def payoff_4(x6,x7,x8,x9):
    return np.maximum(x9-x8,0)
def payoff_5(x5,x6,x7,x8,x9):
    return np.maximum(x9-x8,0)

# Defining the Marginals
for i in range(1,10):
    v.append(np.linspace(100-i,100+i,i+1))
    p.append(np.repeat(1/(i+1),i+1))



def var_function(x,y):
    return ((y/x)**2-1)
min_var = mot(v[7],p[7],v[8],p[8],func=var_function,minimize=True)[0]
max_var = mot(v[7],p[7],v[8],p[8],func=var_function,minimize=False)[0]

variance = np.linspace(min_var,max_var,50)
print(variance)

variance_val = 0.00068

# 2 Marginals
min_w2 = mot(v[7],p[7],v[8],p[8],func=payoff_2,minimize=True,
             variance_info = False, variance_level = variance_val,iterations = 10e20)[0]
max_w2 = mot(v[7],p[7],v[8],p[8],func=payoff_2,minimize=False,
             variance_info = False, variance_level = variance_val,iterations = 10e20)[0]

# 3 Marginals
min_w3 = mot(v[6],p[6],v[7],p[7],v[8],p[8],func=payoff_3 ,minimize=True,variance_info = True, variance_level = variance_val,iterations = 10e20)[0]
max_w3 = mot(v[6],p[6],v[7],p[7],v[8],p[8],func=payoff_3 ,minimize=False,variance_info = True, variance_level = variance_val,iterations = 10e20)[0]

# 4 Marginals
min_w4 = mot(v[5],p[5],v[6],p[6],v[7],p[7],v[8],p[8],func=payoff_4 ,minimize=True,variance_info = True, variance_level = variance_val,iterations = 10e20)[0]
max_w4 = mot(v[5],p[5],v[6],p[6],v[7],p[7],v[8],p[8],func=payoff_4 ,minimize=False,variance_info = True, variance_level = variance_val,iterations = 10e20)[0]

# 5 Marginals
#min_w5 = mot(v[4],p[4],v[5],p[5],v[6],p[6],v[7],p[7],v[8],p[8],func=payoff_5 ,minimize=True,variance_info = True, variance_level = variance_val,iterations =max_iter)[0]
#max_w5 = mot(v[4],p[4],v[5],p[5],v[6],p[6],v[7],p[7],v[8],p[8],func=payoff_5 ,minimize=False,variance_info = True, variance_level = variance_val,iterations =max_iter)[0]


# Plotting:
#max_vector = [max_w2,max_w3,max_w4,max_w5]
#min_vector = [min_w2,min_w3,min_w4,min_w5]

max_vector = [max_w2,max_w3,max_w4]
min_vector = [min_w2,min_w3,min_w4]

plt.scatter(range(3),min_vector, color = "blue", label = "Lower Bound")
plt.scatter(range(3),max_vector, color = "red", label = "Upper Bound")
plt.ylabel('Price Bounds')
plt.xlabel('Number of Intermediate Marginals')
plt.legend(loc="best", frameon=True)
plt.xticks(range(len(max_vector)),fontsize=13)
plt.show()
"""