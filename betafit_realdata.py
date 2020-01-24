#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:03:44 2018

@author: dom
"""

import autograd as ag
import autograd.numpy as np
import autograd.scipy.special as spf
import scipy.optimize as opt


def betaprob(pred_values,lam,nu):
    """log-likelihood of lambda and nu for beta-distributed pred_values. 
    lam=alpha/(alpha+beta), nu=alpha+beta"""
    return np.sum(-spf.betaln(lam*nu,(1.0-lam)*nu)+(lam*nu-1.0)*np.log(pred_values+1e-10)+((1.0-lam)*nu-1.0)*np.log(1.0-pred_values+1e-10))



def time_series_prediction(values,persistence_probs):
    """Likelihood of time series. values[instance,time] \in [0,1] are the pain probability values to be predicted
    persistence_probs: probabilities of pain / nopain persistence
    nu: pseudocount of beta-distribution for likelihood"""
    
    predictions=[]
    for instance in values:
        predicted_values=[]
        for time_point in range(1,len(instance)):
            ppain=instance[time_point-1]
            # Markov chain prediction
            predicted_values.append(ppain*persistence_probs[0]+(1.0-ppain)*(1.0-persistence_probs[1]))
        predictions.append(predicted_values)
        
    return np.array(predictions)
    

def time_series_likelihood(values,persistence_probs,nu):
    """Likelihood of time series. values[instance,time] \in [0,1] are the pain probability values to be predicted
    persistence_probs: probabilities of pain / nopain persistence
    nu: pseudocount of beta-distribution for likelihood"""
    
    predictions=time_series_prediction(values,persistence_probs)

    return np.sum(betaprob(values[:,1:].flatten(),predictions.flatten(),nu))

def cost_function_and_gradient(values,nu):
    
    costfunc=lambda persistence_probs:-time_series_likelihood(values,persistence_probs,nu)
    
    costfunc_grad=ag.grad(costfunc)
    
    return lambda persistence_probs:(costfunc(persistence_probs),costfunc_grad(persistence_probs))


# ground truth. can be used for testing
#pprobs=np.array([0.9,0.3])
#values=[]
#for ni in range(10):
#    instance=[np.random.random()]
#    for tp in range(3):
#        instance.append(instance[-1]*pprobs[0]+(1.0-instance[-1])*(1.0-pprobs[1]))
#    values.append(instance)
#    
#participants=np.array(values)
#    

# Tanja's participants
participants=np.array([[50,20,35,70],[7,5,4,3],[10,10,10,10],
                       [15,50,3,1],[3,3,4,3],[10,5,10,10],
                       [1,1,1,1],[1,3,1,20],[18,33,31,44],
                       [22,35,1,1],[69,73,53,38],[25,30,30,33],
                       [1,1,1,1],[20,45,70,50],[7,13,13,14],
                       [3,3,16,6],[1,50,50,20],[7,4,1,13],[36,20,39,42],
                       [1,32,31,72],[53,38,30,69],[92,48,45,80],[88,83,93,46],
                       [4,82,74,96],[53,73,77,77],[25,56,24,8],
                       [1,1,43,93],[88,91,93,99],[31,75,48,92],
                       [3,31,17,2],[4,13,19,1],[18,44,29,92],
                       [99,99,99,99],[12,43,1,12],[99,54,36,98]])/100.0

# starting values for persistence probabilities of pain and nopain
pprobs=np.array([0.5,0.99])


# iterate participans and fit each one
for values in participants:

    cgf=cost_function_and_gradient(values[np.newaxis,...],1000.0)
    best_p,best_ll,d=opt.fmin_l_bfgs_b(cgf,pprobs,bounds=[(0.01,0.99)]*len(pprobs),factr=1000.0,pgtol=1e-8)
    
    predictions=time_series_prediction(values[np.newaxis,...],best_p)
    
    print("Best transition probabilities",best_p,"Actual and predicted pain values",values[1:],predictions)
    
    
    


