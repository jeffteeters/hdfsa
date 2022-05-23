
import numpy as np


#Parameter to simulate 
# N = 1024  # Hypervector Dimensionality
# D=2**(5) #  size of dictionary with random hypervectors
# K=36 # This is lenght of the sequence, i.e., number of hypervectors to superimpose

# D=2**6

# BERrange = np.arange(0,0.5001,0.05) # Bit Error Rates to explore 
# simul=50 # number of simulations
# simPerBook=10 # simulations per codebook



## Run the functions to compare analytical and empirical
# STAT_ACC_m = AccuracyEmpirical(N,D,K,simul,simPerBook,BERrange)


def bundle_error_empirical(ncols, k, d):
    # convience function to get error with one call
    be = Bundle_empirical(ncols, k, d)
    return be.perr

class Bundle_empirical():

    # calculate bundle (superposition) empirical error

    def __init__(self, ncols, k, d, ntrials=100000, count_multiple_matches_as_error=True, debug=False):
        # ncols is the width of the superposition vector
        # k is number of items stored in sdm
        # d is the number of items in item memory.  Used to compute probability of error in recall with d-1 distractors
        # count_multiple_matches_as_error = True to count multiple distractor hammings == match as error
        self.ncols = ncols
        self.k = k
        self.d = d
        self.ntrials = ntrials
        self.count_multiple_matches_as_error = count_multiple_matches_as_error
        self.debug = debug
        self.empiricalError()

    def empiricalError(self):
        # compute empirical error by storing then recalling items from bundle
        debug = self.debug
        trial_count = 0
        fail_count = 0
        bit_errors_found = 0
        bits_compared = 0
        mhcounts = np.zeros(self.ncols+1, dtype=np.int32)  # match hamming counts
        rng = np.random.default_rng()
        while trial_count < self.ntrials:
            # setup bundle (superpositon)
            hl_cache = {}  # cache mapping address to random hard locations
            contents = np.zeros(self.ncols, dtype=np.int16)  # superpositon vector
            im = rng.integers(0, high=2, size=(self.d, self.ncols), dtype=np.int8) # item memory
            address_base2 = rng.integers(0, high=2, size=(self.k, self.ncols), dtype=np.int8)  # addresses
            # exSeq= np.random.randint(low = 0, high = self.d, size=self.k) 
            exSeq = rng.integers(0, high=self.d, size=self.k, dtype=np.int16) # random sequence to represent
            if debug:
                print("EmpiricalError, trial %s" % (trial_count+1))
                print("im=%s" % im)
                print("address_base2=%s" % address_base2)
                print("exSeq=%s" % exSeq)
                print("contents=%s" % contents)
            # store sequence
            # import pdb; pdb.set_trace()
            for i in range(self.k):
                address = address_base2[i]
                data = im[exSeq[i]]
                vector_to_store = np.logical_xor(address, data)
                contents += vector_to_store*2-1  # convert vector to +1 / -1 then store
                if self.debug:
                    print("Storing item %s" % (i+1))
                    print("address=%s, data=%s, vector_to_store=%s, contents=%s" % (address, data, vector_to_store, contents))
            # recall sequence
            if self.k % 2 == 0:
                # even number items stored, add random vector to break ties
                contents += rng.integers(0, high=2, size=self.ncols, dtype=np.int8) * 2 - 1
            if debug:
                print("Starting recall")
            recalled_vector = contents > 0   # will be binary vector, also works as int8
            for i in range(self.k):
                ## address = address_base[i:i+self.ncols]
                address = address_base2[i]
                data = im[exSeq[i]]
                recalled_data = np.logical_xor(address, recalled_vector)
                hamming_distances = np.count_nonzero(im[:,] != recalled_data, axis=1)
                mhcounts[hamming_distances[exSeq[i]]] += 1
                bit_errors_found += hamming_distances[exSeq[i]]
                hamming_d_found = hamming_distances[exSeq[i]]
                selected_item = np.argmin(hamming_distances)
                if selected_item != exSeq[i]:
                    fail_count += 1
                elif self.count_multiple_matches_as_error:
                    # check for another item with the same hamming distance, if found, count as error
                    hamming_distances[selected_item] = self.ncols+1
                    next_closest = np.argmin(hamming_distances)
                    if(hamming_distances[next_closest] == hamming_d_found):
                        fail_count += 1
                bits_compared += self.ncols
                trial_count += 1
                if trial_count >= self.ntrials:
                    break
                if debug:
                    print("Recall item %s" % (i+1))
                    print("address=%s,data=%s,recalled_vector=%s,recalled_data=%s,hamming_distances=%s,hamming_d_found=%s,fail_count=%s" % (
                        address,data,recalled_vector,recalled_data,hamming_distances,hamming_d_found,fail_count))
                if debug and trial_count > 10:
                    debug=False
        self.perr = fail_count / trial_count  # overall probability of error
        self.mhcounts = mhcounts   # count of hamming distances found to matching item
        self.bit_error_rate = bit_errors_found / bits_compared
        self.ehdist = mhcounts / trial_count  # form distribution of match hammings





## Functions for empirical part
#Estimate Accuracy empirically for given parameters

def AccuracyEmpirical(N,D,K,simul=10000,simPerBook=10,BERrange=[0.0]):
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
    # return probability error rather than probability correct
    error_rate = 1.0 - STAT_ACC_m
    if len(error_rate) == 1:
        # if only one item in vector, convert to scalar (call expecting scalar return if BERrange not specified)
        error_rate = error_rate[0]
    return error_rate

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



    
    # dp_rej=0.5 # Expected dot product for wrong hypervector
    # dp_hit=D/np.sqrt(V) # Expected dot product for correct hypervector
    
    
    # #Calculate BERs
    # BER=[]
    # for eb in range(len(EbNo_range)):
    #     EbNo=EbNo_range[eb] # pick the current EbNo value
    #     SNR= EbNo+10*np.log10(coding_rate) # convert to SNR
    #     noise_var=10**(-SNR/10) # calculate noise variance
        
    #     #Analytical part    
    #     var_hit =(D/2)*(V-1)/V + (D/2)*noise_var # variance for correct hypervector
    #     var_rej= (D/2) + (D/2)*noise_var # variance for wrong hypervector
        
    #     acc_an=p_corr (M, dp_hit, dp_rej, var_hit, var_rej) # get accuracy for the current conditions
    #     ber_an=0.5*(1-acc_an) # get BER from accuracy
        
    #     BER.append(ber_an)
        
    # return BER


    

