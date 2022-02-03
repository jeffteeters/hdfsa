
import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
import pylab
import scipy
import math


## Functions for empirical part


#Representation of sequences
def sequence(itemmem, concepts, N):    
    HDvectors=itemmem[:,concepts]
    
    #prepare rotated HD vectors
    for i in range(0,len(concepts)):
        HDvectors[:,i] =  np.roll(HDvectors[:,i],  -(len(concepts) - 1- i), axis = 0)

    superposition = np.sum(HDvectors, axis=1, keepdims=True)
    k=len(concepts)
    if (k % 2) == 0: # If even number then break ties
        superposition += np.random.randint(low = 0, high = 2, size =(N, 1))
        k+=1
    #print(HDvectors)
    #create composite HD vector
    superposition= (superposition > k/2).astype(int)
    return superposition  # perform bundling of the desired HD vectors    


# Reconstruct the sequence from a given  bundle hypervector 
def reconstructVSA(itemmem, HDcompositional,  K): 
    HDcompositional=2*HDcompositional-1 # turn into bipolar for simplicity
    itemmemT=np.transpose(2*itemmem-1)
    prediction=np.zeros(K)
    for i in range(0,K):
        dp=np.dot(itemmemT,np.roll(HDcompositional, K - 1- i, axis = 0)) # computes dot product
        ind =np.argmax(dp) # get prediction
        prediction[i]= ind
    
    return prediction.astype(int)


#Estimate Accuracy empirically for given parameters
def AccuracyEmpirical(N,D,K,simul=500,simPerBook=10,BERrange=[0]):
    simBooks=int(simul/simPerBook) # number of random codebooks

    STAT_ACC=np.zeros((len(BERrange), simPerBook*simBooks), dtype=float)   

    for iD in range(0,len(BERrange)):
        ber=BERrange[iD]
        # print(ber)
    
        countIter=0 # to count sumilations below
        for simBook in range(0,simBooks):
    
            itemMemory= np.random.randint(low = 0, high = 2, size =(N, D)) # create item memory of a given size
    
            for sim in range(0,simPerBook):
                
                exSeq= np.random.randint(low = 0, high = D, size =K) # radnom sequence to represent
                HDseq=sequence(itemMemory, exSeq, N) # form a compositional hypervector representing the sequence 
                noise=np.random.rand(N, 1)<ber # generate vector of bit flips
                HDseq=np.logical_xor(HDseq,noise).astype(int) # add bitflips to the compositional hypervector
                prediction=reconstructVSA(itemMemory, HDseq,  K) # perform decoding            
                STAT_ACC[iD,countIter]=np.mean(exSeq==prediction) 
                
                countIter+=1


    STAT_ACC_m=np.mean(STAT_ACC, axis=1)
    return STAT_ACC_m
