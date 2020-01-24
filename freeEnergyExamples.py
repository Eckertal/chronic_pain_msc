#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:58:09 2018

@author: dom
"""

from sumProduct import *


# simple node
percept=variableNode("percept",vrange=["pain","tickle"])

expectation=freeEnergyFactorNode()
expectation.addNeighbour(percept)
 
def exprior(condrv,conditioner):
    if condrv[0] is None:
        return 1.0
    if condrv[0]=="tickle":
        return 0.8
    else:
        return 0.2
    
expectation.setPrior(exprior)   
expectation.predictFactor()

allNodes=[percept,expectation]

# marginal prediction at the beginning
runSumProduct(allNodes)
print ("Node ",percept.name,"has marginals",percept.marginal())

# start learning
startLearning(allNodes)
for trial in range(10):
    
    percept.observe("pain")
    # perceive
    runSumProduct(allNodes)
    # remember
    accumulateEvidence(allNodes)
    
learn(allNodes)


# make a prediction
percept.observe()
runSumProduct(allNodes)
print ("Node ",percept.name,"has marginals",percept.marginal())

 
sys.exit()

# simple percept-sense net
percept=variableNode("percept",vrange=["pain","tickle"])
sensory_input=variableNode("sensation",vrange=["nociception","touch"])

expectation=freeEnergyFactorNode()
expectation.addNeighbour(percept)

def exprior(condrv,conditioner):
    if condrv[0] is None:
        return 1.0
    if condrv[0]=="tickle":
        return 0.2
    else:
        return 0.8
    
expectation.setPrior(exprior)
expectation.predictFactor()

prediction=freeEnergyFactorNode()
prediction.addNeighbour(sensory_input)    
prediction.addNeighbour(percept,isConditioner=True)

def predprior(condrv,conditioner):
    print(condrv,conditioner)
    if condrv[0] is None:
        return 10.0
    if conditioner[0]=="pain":
        if condrv[0]=="nociception":
            return 0.9
        else:
            return 0.1
    else:
        if condrv[0]=="nociception":
            return 0.1
        else:
            return 0.9
            
prediction.setPrior(predprior)
prediction.predictFactor()

allVariableNodes=[percept,sensory_input]
allFactorNodes=[expectation,prediction]

allNodes=allVariableNodes+allFactorNodes


#marginal prediction at the beginning
runSumProduct(allNodes)
for node in allVariableNodes:
    print ("Node ",node.name,"has marginals",node.marginal())

# start learning
startLearning(allNodes)
for trial in range(1):
    
    if trial % 2 == 0:
        sensory_input.observe("nociception")
    else:
        sensory_input.observe("touch")
    # perceive
    runSumProduct(allNodes)
    # remember
    accumulateEvidence(allNodes)
    
learn(allNodes)
#    
#    
# make a prediction
sensory_input.observe()
runSumProduct(allNodes)
for node in allVariableNodes:
    print ("Node ",node.name,"has marginals",node.marginal())

  

sys.exit()
    