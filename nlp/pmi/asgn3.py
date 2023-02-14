from load_map import *
from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from wordsim353 import *


def read_counts(filename, wids):
    '''Reads the counts from file. It returns counts for all words, but to
    save memory it only returns cooccurrence counts for the words
    whose ids are listed in wids.

    :type filename: string
    :type wids: list
    :param filename: where to read info from
    :param wids: a list of word ids
    :returns: occurence counts, cooccurence counts, and tot number of observations, and tot number of terms
    '''
    o_counts = {} # Occurence counts
    co_counts = {} # Cooccurence counts
    fp = open(filename)
    N = float(next(fp))
    T = 0 # Term count
    for line in fp:
        T+=1
        line = line.strip().split("\t")
        wid0 = int(line[0])
        o_counts[wid0] = int(line[1])
        if(wid0 in wids):
            co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
    return (o_counts, co_counts, N, T)

def print_sorted_pairs(similarities, o_counts, first=0, last=100, reverse=True):
    '''Sorts the pairs of words by their similarity scores and prints
    out the sorted list from index first to last, along with the
    counts of each word in each pair.

    :type similarities: dict 
    :type o_counts: dict
    :type first: int
    :type last: int
    :param similarities: the word id pairs (keys) with similarity scores (values)
    :param o_counts: the counts of each word id
    :param first: index to start printing from
    :param last: index to stop printing
    :return: none
    '''
    if first < 0: last = len(similarities)
    for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = reverse)[first:last]:
          word_pair = (wid2word[pair[0]], wid2word[pair[1]])
          print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))
            
def freq_v_sim(sims):
    xs = []
    ys = []
    for pair in sims.items():
        ys.append(pair[1])
        c0 = o_counts[pair[0][0]]
        c1 = o_counts[pair[0][1]]
        xs.append(min(c0,c1))
    plt.clf() # clear previous plots (if any)
    plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
    plt.plot(xs, ys, 'k.') # create the scatter plot
    plt.xlabel('Min Freq')
    plt.ylabel('Similarity')
    print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
    plt.show() #display the set of plots

def make_pairs(items):
    '''Takes a list of items and creates a list of the unique pairs
    with each pair sorted, so that if (a, b) is a pair, (b, a) is not
    also included. Self-pairs (a, a) are also not included.

    :type items: list
    :param items: the list to pair up
    :return: list of pairs

    '''
    return [(x, y) for x in items for y in items if x < y]


STEMMER = PorterStemmer()
o_counts,_,_,_ = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", [])
# helper function to get the count of a word (string)
def w_count(word):
    return o_counts[word2wid[word]]

def tw_stemmer(word):
    '''Stems the word using Porter stemmer, unless it is a 
    username (starts with @).  If so, returns the word unchanged.

    :type word: str
    :param word: the word to be stemmed
    :rtype: str
    :return: the stemmed word

    '''
    if word[0] == '@': #don't stem these
        return word
    else:
        return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
    '''Compute the pointwise mutual information using cooccurrence counts.

    :type c_xy: int 
    :type c_x: int 
    :type c_y: int 
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the pmi value

    '''
    return log(c_xy*N/c_x/c_y,2)

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

    
def cos_sim(v0,v1,**kargs):
    '''Compute the cosine similarity between two sparse vectors.

    :type v0: dict
    :type v1: dict
    :param v0: first sparse vector
    :param v1: second sparse vector
    :rtype: float
    :return: cosine between v0 and v1
    '''
    # We recommend that you store the sparse vectors as dictionaries
    # with keys giving the indices of the non-zero entries, and values
    # giving the values at those dimensions.

    dims = sorted(set(list(v0.keys())+list(v1.keys())))
    
    v0_norm = sqrt(np.sum(np.fromiter(v0.values(),float)**2))
    v1_norm = sqrt(np.sum(np.fromiter(v1.values(),float)**2))
    
    v0_arr = np.array([v0[wid] if wid in v0 else 0. for wid in dims])
    v1_arr = np.array([v1[wid] if wid in v1 else 0. for wid in dims])
    
    return v0_arr.dot(v1_arr)/v0_norm/v1_norm

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        vectors[wid0] = {}
        c_wid0 = o_counts[wid0]
        for co_wid in co_counts[wid0]:
            pmi = PMI(co_counts[wid0][co_wid], o_counts[co_wid], c_wid0, tot_count)
            if pmi > 0:
                vectors[wid0][co_wid] = pmi
    return vectors

def PPMI(c_xy, c_x, c_y, N):
    pmi = PMI(c_xy, c_x, c_y, N)
    return pmi if pmi > 0 else None
def create_vectors(function, wids, o_counts, co_counts, tot_count, K=None):
    vectors = {}
    for wid0 in wids:
        vectors[wid0] = {}
        c_wid0 = o_counts[wid0]
        for co_wid in co_counts[wid0]:
            stat = function(co_counts[wid0][co_wid], o_counts[co_wid], c_wid0, tot_count)
            if stat!=None:
                vectors[wid0][co_wid] = stat
        
    if K!=None:
        vectors=reduce(vectors, K)
    return vectors



def compute_similarity(words, col_function, sim_function, print_result=False, K=None, desc=True, **kwargs):
    stemmed_words = [tw_stemmer(w) for w in words]
    all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them
    
    # you could choose to just select some pairs and add them by hand instead
    # but here we automatically create all pairs 
    wid_pairs = make_pairs(all_wids)
    
    #read in the count information
    (o_counts, co_counts, N, T) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)
    
    
    #make the word vectors
    vectors = create_vectors(col_function, all_wids, o_counts, co_counts, N, K)
    
    extra={'T':T}
    if 'gamma' in kwargs:
        extra['gamma'] = kwargs['gamma']
    if 'delta' in kwargs:
        extra['delta'] = kwargs['delta']
        
    
    # compute similarites for all pairs we consider
    sims = {(wid0,wid1): sim_function(vectors[wid0],vectors[wid1],
                                      c1=o_counts[wid0], c2=o_counts[wid1], **extra) for (wid0,wid1) in wid_pairs}
    
    if print_result:
        print("Sort by similarity")
        print_sorted_pairs(sims, o_counts, reverse=desc)

    rank={}
    for i, pair in enumerate(sorted(sims.keys(), key=lambda x: sims[x], reverse = desc)):
        word_pair = (wid2word[pair[0]], wid2word[pair[1]])
        rank[word_pair] = i
    return rank, sims

def reduce(vectors, K):
    reduced = {}
    for wid, vec in vectors.items():
        assert K < len(vec), "K less than vector dimmension {}".format(len(vec))
        threshold = sorted(vec.values(), reverse=True)[K]
        reduced[wid]={}
        for w, v in vec.items():
            if v >= threshold:
                reduced[wid][w]=v
    return reduced

def socpmi_sim(v1, v2, c1, c2, T, gamma=3, delta=6.5):
    beta_1 = int(log(c1)**2*log(T,2)/delta)
    beta_2 = int(log(c2)**2*log(T,2)/delta)
    
    if beta_1>len(v1):
        #print("beta_1({})>len(v1)({}), beta_1 auto adjusted".format(beta_1,len(v1)))
        beta_1=len(v1)
        
    if beta_2>len(v2):
        #print("beta_2({})>len(v2)({}), beta_2 auto adjusted".format(beta_2,len(v2)))
        beta_2=len(v2)
    
    threshold1 = sorted(v1.values(), reverse=True)[beta_1-1]
    threshold2 = sorted(v2.values(), reverse=True)[beta_2-1]
    
    fb1,fb2=0,0
    for w, v in v1.items():
        if v >= threshold1:
            if w in v2:
                fb1+=v2[w]**gamma
    for w, v in v2.items():
        if v >= threshold2:
            if w in v1:
                fb2+=v1[w]**gamma
    
    return fb1/beta_1+fb2/beta_2

def dice(v1, v2, **kwargs):
    s1 = set(v1.keys())
    s2 = set(v2.keys())
    overlap = len(s1 & s2)
    return overlap * 2.0/(len(s1) + len(s2))

def jaccard(v1, v2, **kwargs):
    s1 = set(v1.keys())
    s2 = set(v2.keys())
    union = len(s1 | s2)
    overlap = len(s1 & s2)
    return overlap / union

def spearman_rank_cor(rank1, rank2):
    assert len(rank1)==len(rank2)
    n=len(rank1)
    s = 0
    for w in rank1:
        s+= (rank1[w]-rank2[w])**2
    return 1-6*s/n/(n**2-1)
def min_fre_sim(similarities, min_f=0, max_f=10000000):
    min_freq = []
    sims = []
    for pair, s in similarities.items():
        min_fre = min(o_counts[pair[0]],o_counts[pair[1]])
        if min_fre>=min_f and min_fre<=max_f:
            min_freq.append(min_fre)
            sims.append(s)
    return min_freq, sims
def max_fre_sim(similarities, min_f=0, max_f=10000000):
    max_freq = []
    sims = []
    for pair, s in similarities.items():
        max_fre = max(o_counts[pair[0]],o_counts[pair[1]])
        if max_fre>=min_f and max_fre<=max_f:
            max_freq.append(max_fre)
            sims.append(s)
    return max_freq, sims
def avg_fre_sim(similarities, min_f=0, max_f=10000000):
    avg_freq = []
    sims = []
    for pair, s in similarities.items():
        avg_fre = (o_counts[pair[0]]+o_counts[pair[1]])/2
        if avg_fre>=min_f and avg_fre<=max_f:
            avg_freq.append(avg_fre)
            sims.append(s)
    return avg_freq, sims

def jitter(x, sigma = 1e-7):
    x=np.array(x, dtype=float)
    x+=np.random.randn(x.shape[0])*sigma
    return x
def pearson_cor(x,y):
    return np.cov(x,y)[0,1]/np.std(x)/np.std(y)
def spearman_cor(x,y,jit=False):
    if jit:
        x = jitter(x)
        y = jitter(y)
    rho, p = spearmanr(x,y)
    return rho
    
def spearman_rank_cor(rank1, rank2):
    assert len(rank1)==len(rank2)
    n=len(rank1)
    s = 0
    for w in rank1:
        s+= (rank1[w]-rank2[w])**2
    return 1-6*s/n/(n**2-1)

def plot_cor(sim, size=(8,4)):
    minf,s1 = min_fre_sim(sim)
    maxf,s2 = max_fre_sim(sim)
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=size)
    ax1.plot(minf, s1, '.')
    ax2.plot(maxf, s2, '.')
    
    ax1.set_xlabel("minimum frequency")
    ax2.set_xlabel("maximum frequency")
    ax1.set_ylabel("similarity")
    ax2.set_ylabel("similarity")
    
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    
    ax1.set_title("Similarity & Minimum frequency\npearson:{0:.3f}\nspearman:{1:.3f}".format(pearson_cor(minf,s1),spearman_cor(minf,s1)))
    ax2.set_title("Similarity & Maximum frequency\npearson:{0:.3f}\nspearman:{1:.3f}".format(pearson_cor(maxf,s2),spearman_cor(maxf,s2)))    
    
    fig.tight_layout()

