# -*- coding: utf-8 -*-
"""
Useful Functions
"""
from MOT_2dim2assets import *
import matplotlib.pyplot as plt

def generate_basket_prices(values11,prob11,values12,prob12,values21,prob21,values22,prob22,
                 correlation,basket_strikes,weight_1 = 0.5, weight_2 = 0.5,
                 time = 1,different_Qs = 10):
    
    n = len(values11)*len(values12)*len(values21)*len(values22)

    
    def func1(a,b,c,d):
        return abs(a+c-b-d)
    
    
    if time == 1: # If Basket-Price at time 1 desired
        min_price, min_q = opt_plan_discrete_multiasset(values11,prob11,values12,
                                                        prob12,values21,prob21,
                                                        values22,prob22,func = func1,
                                                        minimize=True,
                                                        correlation_1=True,
                                                        corr_1=correlation)
        max_price, max_q = opt_plan_discrete_multiasset(values11,prob11,values12,
                                                        prob12,values21,prob21,
                                                        values22,prob22,func = func1,
                                                        minimize=False,
                                                        correlation_1=True,
                                                        corr_1=correlation)
        min_q = np.array(min_q)
        max_q = np.array(max_q)
        def basket_payoff(K):
            costs = np.zeros((len(values11),len(values12),len(values21),len(values22)))
            for i in range(len(values11)):
                for j in range(len(values12)):
                    for k in range(len(values21)):
                        for l in range(len(values22)):
                            costs[i,j,k,l] = max(weight_1*values11[i]+weight_2*values12[j]-K,0)
            costs = np.reshape(costs,n)
            return costs
        prices = []
        for j in np.linspace(0,1,different_Qs):
            p = []
            for i in range(len(basket_strikes)):         
                p.append(sum([s*q for s,q in zip(basket_payoff(basket_strikes[i]),min_q*j+max_q*(1-j))]))
            prices.append(p)
        
        
    elif time == 2: # If Basket-Price at time 2 desired
        min_price, min_q = opt_plan_discrete_multiasset(values11,prob11,values12,
                                                        prob12,values21,prob21,
                                                        values22,prob22,func = func1,
                                                        minimize=True,
                                                        correlation_2=True,
                                                        corr_2=correlation)
        max_price, max_q = opt_plan_discrete_multiasset(values11,prob11,values12,
                                                        prob12,values21,prob21,
                                                        values22,prob22,func = func1,
                                                        minimize=False,
                                                        correlation_2=True,
                                                        corr_2=correlation)
        min_q = np.array(min_q)
        max_q = np.array(max_q)
        def basket_payoff(K):
            costs = np.zeros((len(values11),len(values12),len(values21),len(values22)))
            for i in range(len(values11)):
                for j in range(len(values12)):
                    for k in range(len(values21)):
                        for l in range(len(values22)):
                            costs[i,j,k,l] = max(weight_1*values21[k]+weight_2*values22[l]-K,0)
            costs = np.reshape(costs,n)
            return costs
        prices = []
        for j in np.linspace(0,1,different_Qs):
            p = []
            for i in range(len(basket_strikes)):         
                p.append(sum([s*q for s,q in zip(basket_payoff(basket_strikes[i]),min_q*j+max_q*(1-j))]))
            prices.append(p)        
    
    return prices

def prices_to_density(strikes,prices):
    probs = np.zeros(len(strikes))
    for i in range(1,len(strikes)-1):
        probs[i] = (prices[i+1]-2*prices[i]+prices[i-1])/((strikes[i+1]-strikes[i])*(strikes[i]-strikes[i-1]))
    probs[0] = 0
    probs[-1] = (-2*prices[-1]+prices[-2])/((strikes[-1]-strikes[-2])**2)
    #normalize 
    sum_p = sum(probs)
    probs = [p/sum_p for p in probs]
    return strikes, probs

def correlation_from_basket(strikes_1_t1,prices_1_t1,strikes_2_t1,prices_2_t1,strikes_basket,prices_basket):
    s11, p11 = prices_to_density(strikes_1_t1,prices_1_t1)
    s12, p12 = prices_to_density(strikes_2_t1,prices_2_t1)
    sbasket, pbasket = prices_to_density(strikes_basket,prices_basket)
    S_01 = sum([s*p for s,p in zip(s11,p11)])
    S_02 = sum([s*p for s,p in zip(s12,p12)])
    E_mu_11 = sum([(s**2)*p for s,p in zip(s11,p11)])
    E_mu_12 = sum([(s**2)*p for s,p in zip(s12,p12)])
    E_Q = sum([(s**2)*p for s,p in zip(sbasket,pbasket)])
    covariance = (2*E_Q-0.5*E_mu_11-0.5*E_mu_12-S_01*S_02)
    std_dev1 = np.sqrt(E_mu_11-S_01**2)
    std_dev2 = np.sqrt(E_mu_12-S_02**2)
    return covariance/(std_dev1*std_dev2)

def correlation_interval(s11,p11,s12,p12,s21,p21,s22,p22):
    S_01 = sum([s*p for s,p in zip(s11,p11)])
    S_02 = sum([s*p for s,p in zip(s12,p12)])
    E_mu_11 = sum([(s**2)*p for s,p in zip(s11,p11)])
    E_mu_12 = sum([(s**2)*p for s,p in zip(s12,p12)])
    std_dev1 = np.sqrt(E_mu_11-S_01**2)
    std_dev2 = np.sqrt(E_mu_12-S_02**2)
    def f(x_11,x_12,x_21,x_22):
        return (x_11*x_12-S_01*S_02)/(std_dev1*std_dev2)
    min_corr = opt_plan_discrete_multiasset(s11,p11,s12,p12,s21,p21,s22,p22,func=f,onedim=True,minimize=True)[0]
    max_corr = opt_plan_discrete_multiasset(s11,p11,s12,p12,s21,p21,s22,p22,func=f,onedim=True,minimize=False)[0]
    return min_corr, max_corr

def monte_carlo_basket(MC_Nr = 1000,strike_K = 100, sd1 = 0.1, sd2 = 0.2,correlation = 0):
    # Brownian Motion
    def BB_2dim(t=0.1,n=100):
        cov = np.matrix([[(t/n),0],[0,(t/n)]])
        nn= np.random.multivariate_normal([0,0], cov, n)
        B1 = [nn[i][0] for i in range(n)]
        W = [nn[i][1] for i in range(n)]
        B1 = np.cumsum(B1)
        W = np.cumsum(W)
        B2 = [correlation*B1[i]+np.sqrt(1-correlation**2)*W[i] for i in range(n)]
        t1_floor = math.floor((t_1/t)*n)  
        return B1[t1_floor], B2[t1_floor]   
    sigma = np.array([sd1,sd2])
    payoff = []
    
    for i in range(MC_Nr):
      bb1,bb2 = BB_2dim()
      S11, S12 = np.array([S_01,S_02])*np.exp(sigma*np.array([bb1,bb2])-0.5*(sigma**2)*np.array([t_1,t_1]))
      payoff.append(max(0.5*S11+0.5*S12-strike_K,0))

    return np.mean(payoff)
    






# strikes_basket = np.linspace(5,15,100).tolist()
# N=25
# different_Qs = 10
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
# def payoff1(a,b,c,d):
#     return max((1/4)*(a+b+c+d)-10,0)

# def payoff2(a,b,c,d):
#     return (c>a)*(d>b)

# def payoff3(a,b,c,d):
#     return (1/4)*max(c-a,0)*max(d-b,0)

# def payoff4(a,b,c,d):
#     return ((c-a)/a)**2*((d-b)/b)**2

# lower_bound1 = []
# upper_bound1 = []
# lower_bound2 = []
# upper_bound2 = []
# lower_bound3 = []
# upper_bound3 = []
# lower_bound4 = []
# upper_bound4 = []
# lower_bound1_basket = []
# upper_bound1_basket = []
# lower_bound2_basket = []
# upper_bound2_basket = []
# lower_bound3_basket = []
# upper_bound3_basket = []
# lower_bound4_basket = []
# upper_bound4_basket = []

# corr = np.linspace(-0.8713832,0.8713832,N)


# for i in range(N):
#     lower_bound1.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff1,onedim=True,minimize=True,correlation_2=True,corr_2=corr[i])[0])
#     upper_bound1.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff1,onedim=True,minimize=False,correlation_2=True,corr_2=corr[i])[0])
#     lower_bound2.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff2,onedim=True,minimize=True,correlation_2=True,corr_2=corr[i])[0])
#     upper_bound2.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff2,onedim=True,minimize=False,correlation_2=True,corr_2=corr[i])[0])
#     lower_bound3.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff3,onedim=True,minimize=True,correlation_2=True,corr_2=corr[i])[0])
#     upper_bound3.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff3,onedim=True,minimize=False,correlation_2=True,corr_2=corr[i])[0])
#     lower_bound4.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff4,onedim=True,minimize=True,correlation_2=True,corr_2=corr[i])[0])
#     upper_bound4.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,
#                                                      func=payoff4,onedim=True,minimize=False,correlation_2=True,corr_2=corr[i])[0])
#     prices_b = generate_basket_prices(v11,p11,v12,p12,v21,p21,v22,p22,corr[i],strikes_basket,time = 2,different_Qs= different_Qs)
#     l1 = []
#     u1 = []
#     l2 = []
#     u2 = []
#     l3 = []
#     u3 = []
#     l4 = []
#     u4 = []
#     for j in range(different_Qs):
#         l1.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff1,minimize=True,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         u1.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff1,minimize=False,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         l2.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff2,minimize=True,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         u2.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff2,minimize=False,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         l3.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff3,minimize=True,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         u3.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff3,minimize=False,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         l4.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff4,minimize=True,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#         u4.append(opt_plan_discrete_multiasset(v11,p11,v12,p12,v21,p21,v22,p22,payoff4,minimize=False,
#                               basket_prices = prices_b[j],basket_strikes=strikes_basket,basket_time = 2)[0])
#     lower_bound1_basket.append(np.mean(l1))
#     upper_bound1_basket.append(np.mean(u1))
#     lower_bound2_basket.append(np.mean(l2))
#     upper_bound2_basket.append(np.mean(u2))
#     lower_bound3_basket.append(np.mean(l3))
#     upper_bound3_basket.append(np.mean(u3))
#     lower_bound4_basket.append(np.mean(l4))
#     upper_bound4_basket.append(np.mean(u4))
    
# plt.rcParams.update({'font.size': 15})
# fig, axs = plt.subplots(2, 2,figsize = [11,5])
# axs[0, 0].plot(corr, lower_bound1,color="blue")
# axs[0, 0].plot(corr, lower_bound1_basket_max,color="blue",linestyle='dotted')
# axs[0, 0].plot(corr, upper_bound1,color="red")
# axs[0, 0].plot(corr, upper_bound1_basket_min,color="red", linestyle='dotted')
# axs[0, 0].set_title(r'$c_1$')
# axs[0, 0].set_xticklabels([])
# axs[0, 0].set_xticks([-1,-0.5,0,0.5,1])
# axs[0, 0].set_yticks([0.25,0.5,0.75,1])
# axs[0, 0].fill_between(corr, lower_bound1, lower_bound1_basket_max, color='aliceblue')
# axs[0, 0].fill_between(corr, upper_bound1, upper_bound1_basket_min, color='mistyrose')
# axs[0, 1].plot(corr, upper_bound2,color="red")
# axs[0, 1].plot(corr, upper_bound2_basket,color="red", linestyle='dotted')
# axs[0, 1].plot(corr, lower_bound2_basket,color="blue", linestyle='dotted')
# axs[0, 1].plot(corr, lower_bound2,color="blue")
# axs[0, 1].set_title(r'$c_2$')
# axs[0, 1].set_xticklabels([])
# axs[0, 1].set_xticks([-1,-0.5,0,0.5,1])
# axs[0, 1].set_yticks([0,0.2,0.4,0.6])
# axs[0, 1].fill_between(corr, lower_bound2, lower_bound2_basket, color='aliceblue')
# axs[0, 1].fill_between(corr, upper_bound2, upper_bound2_basket, color='mistyrose')
# axs[1, 0].plot(corr, lower_bound3,color="blue")
# axs[1, 0].plot(corr, upper_bound3,color="red")
# axs[1, 0].plot(corr, lower_bound3_basket,color="blue", linestyle='dotted')
# axs[1, 0].plot(corr, upper_bound3_basket,color="red", linestyle='dotted')
# axs[1, 0].set_title(r'$c_3$')
# axs[1, 0].set_xticks([-1,-0.5,0,0.5,1])
# axs[1, 0].set_yticks([0,0.5])
# axs[1, 0].fill_between(corr, lower_bound3, lower_bound3_basket, color='aliceblue')
# axs[1, 0].fill_between(corr, upper_bound3, upper_bound3_basket, color='mistyrose')
# axs[1, 1].plot(corr, upper_bound4,color="red",label = "Correlations, Upper Bound")
# axs[1, 1].plot(corr, upper_bound4_basket,color="red", linestyle='dotted',label = "Basket Prices, Upper Bound")
# axs[1, 1].plot(corr, lower_bound4_basket,color="blue", linestyle='dotted',label = "Basket Prices, Lower Bound")
# axs[1, 1].plot(corr, lower_bound4,color="blue",label = "Correlations, Lower Bound")
# axs[1, 1].set_title(r'$c_4$')
# axs[1, 1].set_xticks([-1,-0.5,0,0.5,1])
# axs[1, 1].set_yticks([0.01,0.02])
# axs[1, 1].fill_between(corr, lower_bound4, lower_bound4_basket, color='aliceblue')
# axs[1, 1].fill_between(corr, upper_bound4, upper_bound4_basket, color='mistyrose')
# axs[0, 0].set(ylabel='Price Bounds')
# axs[1, 0].set(xlabel='Correlation', ylabel='Price Bounds')
# axs[1, 1].set(xlabel='Correlation')
# axs[0,0].grid(True, linestyle='--')
# axs[0,1].grid(True, linestyle='--')
# axs[1,0].grid(True, linestyle='--')
# axs[1,1].grid(True, linestyle='--')


# fig.legend(loc = 7)
# fig.tight_layout()
# fig.subplots_adjust(right=0.65,wspace = 0.25)
# plt.savefig('fig_correlations_improvement_loss_time2.eps', format='eps')
# plt.show()

#plt.tight_layout()
