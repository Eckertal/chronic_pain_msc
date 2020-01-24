######################################################################################
### Example implementation of sum-product algorithm for lecture 5 and exercises 5+6
######################################################################################

import itertools
import functools
import scipy.special as spf
import numpy as np


class node:
    """Node base class. Nodes have neighbour and a name.
    They can send and receive messages."""

    def __init__(self,name):
        self.neighbours=[] # list of neighbours
        self.name=name
        self.clearMessages()

    def clearMessages(self):
        """Clear all received and sent messages"""
        self.receivedMessages=dict() # dictionary {from-name:message content}
        self.sentMessages=dict() # dictionary {to-name:message content}
        


    def addNeighbour(self,neighbour):
        if not isinstance(neighbour,node): 
            raise ValueError("Error adding neighbour: needs to be a node type, but got "+str(neighbour))
        self.neighbours+=[neighbour]
        
    def send(self,msgBuffer,msgType="sum"):
        """Send my messages to the msgBuffer. A message can be computed if messages from all other neighbour have been received.
        The buffer is a list of dictionaries {'from':sender,'to':receiver,'msg':message-content}.
        msgType can be "sum" for sum-product or "max" for max-product
        """
        
        sendableMessages=dict()
        
        for neighbour in self.neighbours:
            
            if neighbour.name in self.sentMessages: 
                continue
            
            allOtherNeighbours=set(self.neighbours)
            allOtherNeighbours.remove(neighbour)
            
            haveAll=True
            for nn in allOtherNeighbours:
                haveAll &= nn.name in self.receivedMessages
            
            if haveAll: sendableMessages.update(self.compute(neighbour,msgType))
            
        msgBuffer+=map(lambda msg:{'from':self.name,'to':msg[0],'msg':msg[1]},sendableMessages.items())
        self.sentMessages.update(sendableMessages)
        


    def receive(self,msgBuffer):
        """Read (and remove) my messages from msgBuffer."""
        for msg in msgBuffer[:]:
            if msg['to']==self.name:
                self.receivedMessages[msg['from']]=msg['msg']
                msgBuffer.remove(msg)

    def done(self):
        """Message-passing iteration is done if this node has sent/received a message to all of its neighbours"""
        allReceived=all(list(map(lambda neighbour:neighbour.name in self.receivedMessages,self.neighbours)))
        allSent=all(list(map(lambda neighbour:neighbour.name in self.sentMessages,self.neighbours)))
        return allReceived & allSent
        
    def compute(self,neighbour,msgType):
        """Compute and return messge to neighbour. May assume that all needed messages have been received"""
        return {neighbour.name:[]}


    # Print methods from here on. 

    def _printNode(self):
        return "node"+self.name+" [shape=octagon,label=\""+self.name+"\"];"

    def _printEdges(self,callingNeighbour=None):
        rv=";\n".join(["node"+self.name+" -- node"+nn.name for nn in filter(lambda x:(callingNeighbour is None) or x.name!=callingNeighbour.name,self.neighbours)])
        if len(rv)>0: rv+=";"
        return rv

    def dotPrint(self,callingNeighbour=None):
        """Print network as dot-file for plotting"""
        if callingNeighbour is None:
            return ("graph factorGraph {\n"
                    +self._printNode()+"\n"
                    +self._printEdges()+"\n"
                    +"".join([n.dotPrint(callingNeighbour=self) for n in self.neighbours])
                    +"}")
        else:
            return (self._printNode()+"\n"
                    +self._printEdges(callingNeighbour)+"\n"
                    +"".join([n.dotPrint(callingNeighbour=self) for n in filter(lambda x:x.name!=callingNeighbour.name,self.neighbours)]))


class variableNode(node):
    """Variable nodes in factor graph. Variables have a range."""

    def __init__(self,name,vrange=[True,False]):
        """By default, variables are boolean"""
        node.__init__(self,name)
        self.vrange=vrange
        self.observe()
        

    def addNeighbour(self,neighbour):
        """Neighbours of variable nodes need to be factor nodes"""
        if not isinstance(neighbour,factorNode): 
            raise ValueError("Error adding neighbour to variable node "+self.name+": need a factor node, but got "+str(neighbour))
        node.addNeighbour(self,neighbour)

    def compute(self,neighbour,msgType):
        """Messages to neighbour: product of all other messages. msgType is only relevant for factor nodes"""
        msg=dict()
        
        allOthers=set(self.neighbours)
        allOthers.remove(neighbour)
        
        for v in self.vrange: 
            msg[v]=functools.reduce(lambda prev,nn:self.receivedMessages[nn.name][v]*prev,allOthers,self.observation[v])
            
        return {neighbour.name:msg}

    def marginal(self,margType="sum"):
        """Returns marginal distribution of this variable, and marginal probability of any observed data"""
        marg=dict()
        for v in self.vrange: 
            marg[v]=functools.reduce(lambda prev,nn:self.receivedMessages[nn.name][v]*prev,self.neighbours,self.observation[v])
        
        if margType=="sum":
            dataMarg=sum(list(marg.values()))
            for v in self.vrange: marg[v]/=dataMarg
            return marg,dataMarg
            
        elif margType=="max":
            maxMarg=0.0
            maxArg=None
            for v in self.vrange:
                if marg[v]>maxMarg:
                    maxMarg=marg[v]
                    maxArg=v
            return maxMarg,maxArg
            
        else: raise ValueError("Unimplemented marginal type "+str(margType))
            
        

    def observe(self,value=None):
        if value is None: # make variable unobserved
            self.observation=dict((v,1) for v in self.vrange)
            return
        if value not in self.vrange: raise ValueError("Cannot set variable "+self.name+" with range "+str(self.vrange)+" to value "+str(value))
        self.observation=dict((v,0.0) for v in self.vrange)
        self.observation[value]=1.0
                                                        

    def _printNode(self):
        return "node"+self.name+" [shape=circle,label=\""+self.name+"\"];"
    
            
            
class factorNode(node):
    """Factor node in factor graph."""

    def __init__(self,name="factor"):
        node.__init__(self,name)
        self.ranges=None
        self.factor=dict()

    def addNeighbour(self,neighbour):
        """Neighbours of factor nodes need to be variable nodes"""
        if not isinstance(neighbour,variableNode): raise ValueError("Error adding neighbour to factor node "+self.name+": need a variable node, but got "+str(neighbour))
        node.addNeighbour(self,neighbour)
        neighbour.addNeighbour(self)
        
        self.name+=neighbour.name # name of the factor node: concatenation of names of neighbouring variable nodes
        
        # range of the factor's arguments is built up when new neighbour is added
        if self.ranges is None: self.ranges=[(x,) for x in neighbour.vrange]
        else: self.ranges=[x+(y,) for x,y in itertools.product(self.ranges,neighbour.vrange)]

    def compute(self,neighbour,msgType):
        """Messages to neighbours: product of all other messages times factor"""
        
        msg=dict((v,0.0) for v in neighbour.vrange)
        nidx=self.neighbours.index(neighbour)

        allOthers=self.neighbours[:nidx]+self.neighbours[nidx+1:]        
        
        for vr in self.ranges:

            vrao=vr[:nidx]+vr[nidx+1:]            
            
            msgContrib=functools.reduce(lambda prev,nnv:self.receivedMessages[nnv[0].name][nnv[1]]*prev,
                                                     zip(allOthers,vrao),
                                                     self.factor[vr])
                                                     
                                                     
            if msgType=="sum": msg[vr[nidx]]+=msgContrib
            elif msgType=="max": msg[vr[nidx]]=max(msg[vr[nidx]],msgContrib)
            else: raise RuntimeError("Unknown message type:"+str(msgType))
            
        return {neighbour.name:msg}
                                                             
    def setValues(self,callback):
        """Get the values for the factor by calling callback, a function which accepts a tuple of values as argument"""
        for vr in self.ranges: self.factor[vr]=callback(vr)

    def _printNode(self):
        return "node"+self.name+" [shape=box,label=\""+self.name+"\"];"
        
        
class freeEnergyFactorNode(factorNode):
    """Factor node with an exponential family prior on conditional multinomial distributions"""
    
    def __init__(self):
        factorNode.__init__(self,"freeEnergyFactor:|")
        self.conditionerRanges=None
        self.conditionedRanges=None
        self.factor=dict()
        self.pseudoCounts=dict()
        self.naturalParams=dict()
        
        self.accumulatedResponsibilities=dict()
        
        
    def addNeighbour(self,neighbour,isConditioner=False):
        """Neighbours of factor nodes need to be variable nodes. If isConditioner is True, then this neighbour appears behind the conditioning bar in the corresponding probability factor"""
        if not isinstance(neighbour,variableNode): raise ValueError("Error adding neighbour to free energy factor node "+self.name+": need a variable node, but got "+str(neighbour))
        node.addNeighbour(self,neighbour)
        neighbour.addNeighbour(self)
        
        cpos=self.name.find("|")
        if isConditioner:
            self.name+=neighbour.name
            if self.conditionerRanges is None: 
                self.conditionerRanges=[(x,) for x in neighbour.vrange]
            else:
               self.conditionerRanges=[x+(y,) for x,y in itertools.product(self.conditionerRanges,neighbour.vrange)] 
        else:
            self.name=self.name[:cpos]+neighbour.name+self.name[cpos:]
            if self.conditionedRanges is None: 
                self.conditionedRanges=[(x,) for x in neighbour.vrange]
            else:
               self.conditionedRanges=[x+(y,) for x,y in itertools.product(self.conditionedRanges,neighbour.vrange)] 
               
        
    
    def resetResponsibilities(self):
        """Reset all accumulated responsibilities to 0 and KL(post|prior)=0, i.e. restart learning"""
        
        for conditioner in self.conditionerRanges:
            for condrv in self.conditionedRanges:
                self.accumulatedResponsibilities[(condrv,conditioner)]=0.0
                
        self.KLDiv_post_prior=0.0
    
    def setPrior(self,callback):
        """Get values for the factor's exponential family prior by calling callback(conditioned-values,condititoner-values).
        if conditioned-values are supplied, callback needs to return the natural parameters for the supplied conditioners. 
        if conditioned-values are (None,), callback needs to return the pseudocount for the supplied conditioners.
        conditioner tuple is (None,) if this is a root node"""
        
        self.naturalParams=dict()
        self.pseudoCounts=dict()
        
         # special case: this is a root note. need one default conditioner
        if self.conditionerRanges is None:
            self.conditionerRanges=[(None,)]
            self.ranges=self.conditionedRanges
        else:
            # range of the factor's arguments can be computed when order of conditioned/conditioner variables is known, which must be fixed before prior is set
            self.ranges=[c+d for c,d in itertools.product(self.conditionedRanges,self.conditionerRanges)]


        for conditioner in self.conditionerRanges:
            self.pseudoCounts[conditioner]=callback((None,),conditioner)
            for condrv in self.conditionedRanges:
                self.naturalParams[(condrv,conditioner)]=callback(condrv,conditioner)
                
        self.resetResponsibilities()
        
       
        
        
    def accumulate(self):
        """Accumulate current factor information into sufficient statistics and counts. Call after each datapoint"""
        
        # check if we have all the messages we need to compute responsibilities
        haveAll=True
        for nn in self.neighbours:
            haveAll &= nn.name in self.receivedMessages
        
        if not haveAll:
            return False
         
        
        margProb=0.0
        curJointProb=dict()
        for conditioner in self.conditionerRanges:
            for condrv in self.conditionedRanges:
                
                
                vr=condrv
                if conditioner != (None,):
                    vr=vr+conditioner
                
               
                curJointProb[(condrv,conditioner)]=functools.reduce(lambda prev,nnv:self.receivedMessages[nnv[0].name][nnv[1]]*prev,
                                                                            zip(self.neighbours,vr),
                                                                            self.factor[vr])                                                    
                margProb+=curJointProb[(condrv,conditioner)]
                
        
        for conditioner in self.conditionerRanges:
            for condrv in self.conditionedRanges:
                 self.accumulatedResponsibilities[(condrv,conditioner)]+=curJointProb[(condrv,conditioner)]/margProb
                
        return True
        

    def predictFactor(self):
        """Predict expected factor values from prior parameters"""
        for conditioner in self.conditionerRanges:
            
            nu=self.pseudoCounts[conditioner]
            
            fsum=0.0
            
            ccond=tuple()
            if conditioner != (None,):
                ccond=conditioner
    
            for condrv in self.conditionedRanges:
                self.factor[condrv+ccond]=np.exp(spf.psi(nu*self.naturalParams[(condrv,conditioner)])-spf.psi(nu))
                fsum+=self.factor[condrv+ccond]
            for condrv in self.conditionedRanges:
                self.factor[condrv+ccond]/=fsum
                
                
    def _logf(self,lam,nu):
        """Log of normalization constant"""
        return spf.gammaln(nu)-sum(spf.gammaln(lam*nu))
            
            
    def _dlogfdnu(self,lam,nu):
        """Log of derviative of normalization constant"""
        return spf.psi(nu)-sum(lam*spf.psi(lam*nu))
        
    def _etaexp(self,lam,nu):
        """Expectation of eta, the real natural parameter"""
        return lam[:-1]*(spf.psi(nu*lam[:-1])-spf.psi(nu*lam[-1]))
        
    def updatePriorToPosterior(self):
        """Update the prior parameters to posterior with the accumulated responsibilities"""
        naturalParams_new=dict()
        for conditioner in self.conditionerRanges:
            
            nu_old=self.pseudoCounts[conditioner]
            nu_new=nu_old
            
            ccond=tuple()
            if conditioner != (None,):
                ccond=conditioner
    
            
            
            natPar=[]
            natPar_new=[]
            for condrv in self.conditionedRanges:
                naturalParams_new[(condrv,conditioner)]=nu_old*self.naturalParams[(condrv,conditioner)]+self.accumulatedResponsibilities[(condrv,conditioner)]
                nu_new+=self.accumulatedResponsibilities[(condrv,conditioner)]
            for condrv in self.conditionedRanges:
                naturalParams_new[(condrv,conditioner)]/=nu_new
                natPar_new.append(naturalParams_new[(condrv,conditioner)])
                natPar.append(self.naturalParams[(condrv,conditioner)])
                
            
            natPar=np.array(natPar)
            natPar_new=np.array(natPar)
            
            self.KLDiv_post_prior+=self._logf(natPar_new,nu_new)-self._logf(natPar_new,nu_old)-(nu_new-nu_old)*self._dlogfdnu(natPar_new,nu_new)+nu_old*np.dot(self._etaexp(natPar_new,nu_new),(natPar_new-natPar)[:-1])
            self.pseudoCounts[ccond]=nu_new
            
        self.naturalParams=naturalParams_new
        
            
            

    
def runSumProduct(allNodes,msgType="sum"):
    """Run sum-product on allNodes"""
    print("Runing Sum-product")
    done=False
    msgBuffer=[]
    for no in allNodes: no.clearMessages()
    while not done:
        done=True
        for no in allNodes:
            no.receive(msgBuffer)
            no.send(msgBuffer,msgType)
            done &= no.done()
    
        
def startLearning(allNodes):
    print("starting learning for",len(allNodes),"nodes")
    for node in allNodes:
        if isinstance(node,freeEnergyFactorNode):
            node.predictFactor()
            node.resetResponsibilities()
        
def accumulateEvidence(allNodes):
    print("accumulating evidence for",len(allNodes),"nodes")
    for node in allNodes:
        if  isinstance(node,freeEnergyFactorNode):
            node.accumulate()
            
        
def learn(allNodes):
    print("learning for",len(allNodes),"nodes")
    for node in allNodes:
        if isinstance(node,freeEnergyFactorNode):
            node.updatePriorToPosterior()
            node.predictFactor()
        
        

            
    
if __name__ == "__main__":

    # to render the graphs, graphiviz package needs to be installed
    

    # simple markov chain
    # variable nodes
    variables=[variableNode(name,vrange=[0,1]) for name in "ABCD"]
    
    # factor nodes
    factors=[]
    # prior on A
    pA=factorNode()
    pA.addNeighbour(variables[0])
    pA.setValues(lambda x:0.3+0.4*x[0])
    factors+=[pA]
    # conditional factors
    parent=variables[0]
    for child in variables[1:]:
        p=factorNode()
        p.addNeighbour(parent)
        p.addNeighbour(child)
        p.setValues(lambda x:0.2*(x[0]==x[1])+0.4)
        factors+=[p]
        parent=child


    allNodes=variables+factors
    
    # compute node marginals
    runSumProduct(allNodes)
    for v in variables: 
        print ("Node ",v.name,"has marginals",v.marginal())
    print

    # observe last node. compute conditionals
    variables[-1].observe(0)
    runSumProduct(allNodes)
    print ("Last node observed")
    for v in variables: 
        print ("Node ",v.name,"has marginals",v.marginal())
    variables[-1].observe() # un-observe last node
    

    # maximally probable hidden state sequence, no observed nodes
    runSumProduct(allNodes,"max")
    print()
    print("Maximally probable state sequence, no observed nodes")
    for v in variables: 
        print(v.name+" has max marginals ",v.marginal("max"))

    # maximally probable hidden state sequence, no observed nodes

    variables[-1].observe(0)
    runSumProduct(allNodes,"max")
    print()
    print ("Maximally probable state sequence, last node observed")
    for v in variables: 
        print(v.name+" has max marginals ",v.marginal("max"))
        
        


    
    
     
 
    


  
    
    
    
    
    
    
    
    
