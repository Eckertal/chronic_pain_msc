# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:54:38 2017

@author: OEM
"""
import numpy as np
import matplotlib.pyplot as plt

#Use a list because the deck is ordered
def makeDeckAndShuffle(stacked=False):
    deck=[]
    if stacked:
        faces=[2,3,4,5,6,7,2,9,10,"jack","queen","king","ace"]
    else:
        faces=[2,3,4,5,6,7,8,9,10,"jack","queen","king","ace"]
    for suit in ["clubes","hearts","spades","diamonds"]:
        for face in faces:
            deck.append((face,suit))
            
    np.random.shuffle(deck)         
    return deck

def facesToValues(card):
    face,suit=card
    if face in ["jack","queen","king"]:
        return 10
    if face=="ace":
        return 11
    return int(face)

def blackjack(cards):
    sumOfvalues=0
    for card in cards:
        sumOfvalues+=facesToValues(card)
        
    return sumOfvalues==21

def runGame(stacked=False):
    numOfAcesOnFirstDraw=0.0
    numOfAcesOnFirstDrawAndBlackJack=0.0
    for numberOfGame in range(10000): 
        currentGame=makeDeckAndShuffle(stacked)
        draws=currentGame[:3]
        if draws[0][0]=="ace":
            numOfAcesOnFirstDraw+=1.0
            
        if draws[0][0]=="ace" and blackjack(draws):
            numOfAcesOnFirstDrawAndBlackJack+=1.0
        
    pOfAcesOnFirstDraw=numOfAcesOnFirstDraw/(numberOfGame+1.0)
    pOfAcesOnFirstDrawAndBlackJack=numOfAcesOnFirstDrawAndBlackJack/(numberOfGame+1.0)
    pOfBlackJackGivenAceOnFirstDraw=pOfAcesOnFirstDrawAndBlackJack/pOfAcesOnFirstDraw
    return pOfBlackJackGivenAceOnFirstDraw

fairScores=[]
for runs in range(100):
    fairScores+=[runGame(False)]
    
stackedScores=[]
for runs in range(100):
    stackedScores+=[runGame(True)]    

    
fairScores=np.array(fairScores)
print("mean and std deviation of fair scores", fairScores.mean(),np.sqrt(fairScores.var()))

stackedScores=np.array(stackedScores)
print("mean and std deviation of stacked scores", stackedScores.mean(), np.sqrt(stackedScores.var()))


plt.hist(fairScores,bins=10,normed=True,facecolor="blue",label="FAIR")
plt.hist(stackedScores, bins=10, normed=True, alpha=0.5, facecolor="red",label="STACKED")
plt.legend()
plt.show()

