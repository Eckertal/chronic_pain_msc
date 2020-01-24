# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:57:10 2017

@author: dom
"""

import numpy
import matplotlib.pyplot as plt

def makeDeckAndShuffle(stacked=False):
    deck=[]
    if stacked:
        faces=[2,3,4,5,6,7,2,9,10,"jack","queen","king","ace"]
    else:
        faces=[2,3,4,5,6,7,8,9,10,"jack","queen","king","ace"]
    for suit in ["clubs","hearts","spades","diamonds"]:
        for face in faces:
            deck.append((face,suit))
            
    numpy.random.shuffle(deck)
    return deck

def facesToValues(card):
    face,suit=card
    if face in ["jack","queen","king"]: 
        return 10
    if face=="ace":
        return 11
    return int(face)
    
def blackjack(cards):
    sumOfValues=0
    for card in cards:
        sumOfValues+=facesToValues(card)
    haveBlackjack=sumOfValues==21
    return haveBlackjack


def runGame(stacked=False,numGames=1000):
    numOfAcesOnFirstDraw=0.0
    numOfAcesOnFirstDrawAndBlackJack=0.0  
    numOfBlackJacks=0.0
    
    for numberOfGame in range(numGames):
        currentGame=makeDeckAndShuffle(stacked)
        draws=currentGame[:3]
        
        if draws[0][0]=="ace":
            numOfAcesOnFirstDraw+=1.0
     
        if draws[0][0]=="ace" and blackjack(draws):
            numOfAcesOnFirstDrawAndBlackJack+=1.0 
            
        if blackjack(draws):
            numOfBlackJacks+=1.0
    
        
    pOfAcesOnFirstDraw=numOfAcesOnFirstDraw/(numberOfGame+1.0)
    pOfAcesOnFirstDrawAndBlackJack=numOfAcesOnFirstDrawAndBlackJack/(numberOfGame+1.0)
    pOfBlackJackGivenAceOnFirstDraw=pOfAcesOnFirstDrawAndBlackJack/pOfAcesOnFirstDraw
    return pOfBlackJackGivenAceOnFirstDraw
    

fairScores=[] 
for runs in range(1000):
    fairScores+=[runGame(False)]

stackedScores=[] 
for runs in range(1000):
    stackedScores+=[runGame(True)]

allScores=fairScores+stackedScores
numBins=5
mhist,mbins=numpy.histogram(allScores,numBins)

mhist=numpy.array(mhist)/float(len(allScores))

#marginal dist
plt.bar(mbins[:-1],mhist,mbins[1:]-mbins[:-1])
plt.show()

#Likelihood for the hypothesis that the deck is stacked...
stackedHist,stackedBins=numpy.histogram(stackedScores,mbins)
stackedHist=numpy.array(stackedHist)/float(len(stackedScores))

stackedPosterior=stackedHist+0.5/mhist

#likelihood of stacked Hyp
plt.bar(mbins[:-1],stackedHist,mbins[1:]-mbins[:-1])
plt.show()

#Posterior stacked Hyp
plt.bar(mbins[:-1],stackedPosterior,mbins[1:]-mbins[:-1])
plt.show()

        

#Likelihood for the hypothesis that the deck is fair...
#fairHist, fairBins = numpy.histogram(fairScores,mbins)
#fairHist=numpy.histogram(fairScores,mbins)
    
#fairScores=numpy.array(fairScores)
#print("Fair game",fairScores.mean(),numpy.sqrt(fairScores.var()))

#stackedScores=numpy.array(stackedScores)
#print("Stacked game",stackedScores.mean(),numpy.sqrt(stackedScores.var()))


#n,bins,patches=plt.hist(fairScores,10,normed=True,facecolor="blue",alpha=0.5)
#plt.hist(stackedScores,10,normed=True,facecolor="red",alpha=0.5)

#plt.show()
