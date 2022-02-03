#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script was provided by Denis Kleyko.  He describes it as:
# In 2017 paper “High-Dimensional Computing as a Nanoscalable Paradigm” (see page 8)
# we gave equations that allow computing the accuracy of retrieval from the hypervector
# storing a sequence of K symbols from an alphabet of size D in N-dimensional binary 
# hypervector that might also have been exposed to certain amount of bit flips. 
# The process also uses the equation Pentti's "Fully Distributed Representation"
# Attached is the python code that makes the illustration for both equations and for
# numerical simulations.

# I added code to also display the accuracy using the equation derived from the paper:
# Gallant, S. I. and Okaywe, T. W. (2013). Representing Objects, Relations, and
# Sequences. Neural Computation 25, 2038–2078 [in press: Aug. 2013]
# The Gallant equation does not match that Frady equation implemented by the code Dennis
# provided.
    

import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
import pylab
import scipy
import math


#Parameter to simulate 
N = 1024  # Hypervector Dimensionality
D=2**(5) #  size of dictionary with random hypervectors
K=36 # This is lenght of the sequence, i.e., number of hypervectors to superimpose

# D=2**6

BERrange = np.arange(0,0.5001,0.05) # Bit Error Rates to explore 
simul=50 # number of simulations
simPerBook=10 # simulations per codebook



## Functions for analytical part

# Calculate analytical accuracy of the encoding according to the equation for p_corr from 2017 IEEE Tran paper
def  p_corr (N, D, dp_hit):
    dp_rej=0.5
    var_hit = 0.25*N
    var_rej=var_hit
    range_var=10 # number of std to take into account
    fun = lambda u: (1/(np.sqrt(2*np.pi*var_hit)))*np.exp(-((u-N*(dp_rej-dp_hit) )**2)/(2*(var_hit)))*((norm.cdf((u/np.sqrt(var_rej))))**(D-1) ) # analytical equation to calculate the accuracy  

    acc = integrate.quad(fun, (-range_var)*np.sqrt(var_hit), N*dp_rej+range_var*np.sqrt(var_hit)) # integrate over a range of possible values
    print("p_corr, N=%s, D=%s, dp_hit=%s, return=%s" % (N, D, dp_hit, acc[0]))
    return acc[0]


#Compute expected Hamming distance
def  expectedHamming (K):
    if (K % 2) == 0: # If even number then break ties so add 1
        K+=1    

    deltaHam = 0.5 - (scipy.special.binom(K-1, 0.5*(K-1)))/2**K  # Pentti's formula for the expected Hamming distance
    return deltaHam



#Estimate accuracy analytically for given parameters
def AccuracyAnalytical(N,D,K,BERrange):

    EST_ACC=np.zeros(len(BERrange), dtype=float)   

    for iD in range(0,len(BERrange)):
        ber=BERrange[iD]
        #print(ber)
        deltaHam=  expectedHamming (K) #expected Hamming distance
        deltaBER=(1-2*deltaHam)*ber # contribution of noise to the expected Hamming distance
        dp_hit= deltaHam+deltaBER # total expected Hamming distance
        print("in AccuracyAnalytical, K=%s,  ber=%s, deltaHam=%s, deltaBER=%s, dp_hit=%s" %(
            K, ber, deltaHam, deltaBER, dp_hit))

        EST_ACC[iD]=p_corr (N, D, dp_hit) # expected accuracy

    return EST_ACC


#Estimate accuracy analytically for given parameters using Gallant equations
def AccuracyAnalyticalGallant(N,D,K,BERrange):

    EST_ACC=np.zeros(len(BERrange), dtype=float)   

    for iD in range(0,len(BERrange)):
        ber=BERrange[iD]
        #print(ber)
        deltaHam=  expectedHamming (K) #expected Hamming distance
        deltaBER=(1-2*deltaHam)*ber # contribution of noise to the expected Hamming distance
        dp_hit= deltaHam+deltaBER # total expected Hamming distance

        EST_ACC[iD]=p_corr_Gallant(N, D, dp_hit) # expected accuracy

    return EST_ACC


# Calculate analytical accuracy using method in Gallant paper
def p_corr_Gallant(N, D, dp_hit):
    match_mean = dp_hit
    match_stdev = math.sqrt(match_mean * (1-match_mean)/N)
    distractor_mean = 0.5
    distractor_stdev = math.sqrt(distractor_mean *(1-distractor_mean)/N)
    combined_mean = distractor_mean - match_mean
    combined_stdev = math.sqrt(match_stdev**2 + distractor_stdev**2)
    perror1 = norm.cdf(0.0, loc=combined_mean, scale=combined_stdev)
    pcorrect_1 = 1.0 - perror1
    pcorrect_n = pcorrect_1 ** (D-1)
    return pcorrect_n


## Functions for empirical part


#Representation of sequences
def sequence(itemmem, concepts):    
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
def AccuracyEmpirical(N,D,K,simul,simPerBook,BERrange):
    simBooks=int(simul/simPerBook) # number of random codebooks

    STAT_ACC=np.zeros((len(BERrange), simPerBook*simBooks), dtype=float)   

    for iD in range(0,len(BERrange)):
        ber=BERrange[iD]
        print(ber)
    
        countIter=0 # to count sumilations below
        for simBook in range(0,simBooks):
    
            itemMemory= np.random.randint(low = 0, high = 2, size =(N, D)) # create item memory of a given size
    
            for sim in range(0,simPerBook):
                
                exSeq= np.random.randint(low = 0, high = D, size =K) # radnom sequence to represent
                HDseq=sequence(itemMemory, exSeq) # form a compositional hypervector representing the sequence 
                noise=np.random.rand(N, 1)<ber # generate vector of bit flips
                HDseq=np.logical_xor(HDseq,noise).astype(int) # add bitflips to the compositional hypervector
                prediction=reconstructVSA(itemMemory, HDseq,  K) # perform decoding            
                STAT_ACC[iD,countIter]=np.mean(exSeq==prediction) 
                
                countIter+=1


    STAT_ACC_m=np.mean(STAT_ACC, axis=1)
    return STAT_ACC_m


    
    dp_rej=0.5 # Expected dot product for wrong hypervector
    dp_hit=D/np.sqrt(V) # Expected dot product for correct hypervector
    
    
    #Calculate BERs
    BER=[]
    for eb in range(len(EbNo_range)):
        EbNo=EbNo_range[eb] # pick the current EbNo value
        SNR= EbNo+10*np.log10(coding_rate) # convert to SNR
        noise_var=10**(-SNR/10) # calculate noise variance
        
        #Analytical part    
        var_hit =(D/2)*(V-1)/V + (D/2)*noise_var # variance for correct hypervector
        var_rej= (D/2) + (D/2)*noise_var # variance for wrong hypervector
        
        acc_an=p_corr (M, dp_hit, dp_rej, var_hit, var_rej) # get accuracy for the current conditions
        ber_an=0.5*(1-acc_an) # get BER from accuracy
        
        BER.append(ber_an)
        
    return BER


    
    




## Run the functions to compare analytical and empirical
STAT_ACC_m = AccuracyEmpirical(N,D,K,simul,simPerBook,BERrange)
EST_ACC = AccuracyAnalytical(N,D,K,BERrange)
EST_ACC2 = AccuracyAnalyticalGallant(N,D,K,BERrange)


#Plot results 

plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(9,6))

plt.plot(BERrange, 1-STAT_ACC_m, label="Empirical", color= 'r',  linestyle = 'dashed', lw=2)
plt.plot(BERrange, 1-EST_ACC, label="Analytical", color= 'b',  linestyle = 'solid', lw=2)

plt.plot(BERrange, 1-EST_ACC2, label="Analytical Gallant", color= 'g',  linestyle = 'solid', lw=2)

plt.yscale('log')
plt.legend(loc="lower left")
plt.xlabel("Bit Error Rate")
plt.ylabel("Accuracy")
#plt.xticks([i for i in range(0,dimCodebook )])  
plt.grid()
plt.show()

