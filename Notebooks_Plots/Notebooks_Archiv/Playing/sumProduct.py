######################################################################################
### Example implementation of sum-product algorithm for lecture 5 and exercises 5+6
######################################################################################

import itertools
import functools


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
        allReceived=all(map(lambda neighbour:neighbour.name in self.receivedMessages,self.neighbours))
        allSent=all(map(lambda neighbour:neighbour.name in self.sentMessages,self.neighbours))
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
            dataMarg=sum(marg.values())
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
        if value not in self.vrange: raise self.valueError("Cannot set variable "+self.name+" with range "+str(self.vrange)+" to value "+str(value))
        self.observation=dict((v,0.0) for v in self.vrange)
        self.observation[value]=1.0
                                                        

    def _printNode(self):
        return "node"+self.name+" [shape=circle,label=\""+self.name+"\"];"
    
            
            
class factorNode(node):
    """Factor node in factor graph."""

    def __init__(self):
        node.__init__(self,'factor')
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


    
def runSumProduct(allNodes,msgType="sum"):
    """Run sum-product on allNodes"""
    done=False
    msgBuffer=[]
    for no in allNodes: no.clearMessages()
    #for n in range(20):
    while not done:
        done=True
        for no in allNodes:
            no.receive(msgBuffer)
            no.send(msgBuffer,msgType)
            #print(no)
            done &= no.done()
    
        
    
if __name__ == "__main__":

    # to render the graphs, graphviz package needs to be installed

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

    # maximally probable hidden state sequence, last node observed?

    variables[-1].observe(0)
    runSumProduct(allNodes,"max")
    print()
    print ("Maximally probable state sequence, last node observed")
    for v in variables: 
        print(v.name+" has max marginals ",v.marginal("max"))
        
        


