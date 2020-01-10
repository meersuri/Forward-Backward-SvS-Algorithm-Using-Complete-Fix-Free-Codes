# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:20:28 2019

@author: Meer Suri
"""

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from itertools import product, combinations
import pickle

def binarySearchNext(postingList, low, high, current):
    comps = 0
    while high-low > 1:
        mid = (low + high)//2
        if postingList[mid] <= current:
            low = mid
            comps += 1
        else:
            high = mid
            comps += 1
    return low, comps

#def gallopingSearchNext(postingList, l, h, current):
#    comps = 1
#    low = l
#    jump = 1
#    high = low + jump
#    while high < h and postingList[high] <= current:
#        low = high
#        jump = 2*jump
#        high = low + jump
#        comps += 1
#    if high > h:
#        high = h
#        comps += 1
#    pos, cost = binarySearchNext(postingList, low, high, current)
#    comps += cost
#    return pos, comps

def toBin(n,width):
    x = bin(n)[2:]
    x = x.zfill(width)
    l = len(x)
    b = np.zeros(l,dtype='int32')
    for i in range(l):
        if x[i] == '1':
            b[i] = 1
        else:
            b[i] = 0
    return b
          
            
def buildIndex(source, codeBook):
    N = len(source)
    # postings dictionary
    dictionary = {}  

    # init postings dictionary    
    for code in codeBook:
        dictionary[code] = []  
        
#     fill dictionary     
    i = 0
    while i<N:
        j = 1
        while (i+j <= N  and not source[i:i+j] in dictionary):
            j = j+1
        if i+j <= N:
            dictionary[source[i:i+j]].append(i)
            i = i + 1
        else:
            break
                 
        
    return dictionary


def generateSource(p,N):
    X = np.random.binomial(1,p,N)  # sequence
    Xs = ''.join(map(str, X))
    return Xs

def queryIndexFwd(phraseGrams, dictionary, source):
    
    n = len(phraseGrams)
    listLens = []
    matchPos = []
    codeLens = []
    totalComps = 0
    for i in range(n):
        l = len(dictionary[phraseGrams[i]])
        listLens.append(l)
        codeLens.append(len(phraseGrams[i]))
#    for code in phraseGrams:
#        codeLens.append(len(code))
    indexes = np.argsort(listLens)
    rarestTerm = phraseGrams[indexes[0]]
    delta = indexes[0]
            
    candidates = list.copy(dictionary[rarestTerm])
    for i in range(1, n):  
        currentTerm = phraseGrams[indexes[i]]
        delta = indexes[0] - indexes[i]
        
        filtered = []
        
        if delta > 0:
            gap = np.sum(codeLens[indexes[i]:indexes[0]])
        else:
            gap = -np.sum(codeLens[indexes[0]:indexes[i]])
            
        for j in range(len(candidates)):

            val = candidates[j] - gap
#            x, searchComps = binarySearchNext(dictionary[currentTerm],
#                                              0, len(dictionary[currentTerm]),
#                                              val)
            searchComps = np.ceil(np.log2(listLens[indexes[i]]))
            totalComps += searchComps
#            if (dictionary[currentTerm][x] == val):
#                filtered.append(candidates[j])
            if (source[val: val + codeLens[indexes[i]]] == currentTerm):
                filtered.append(candidates[j])
            totalComps += 1
        
        candidates = list.copy(filtered)

    matchPos = np.array(filtered) - (np.sum(codeLens) - np.sum(codeLens[indexes[0]:]))
                           
    return matchPos, totalComps

def queryIndexBwd(phraseGrams, dictionary, source):
    n = len(phraseGrams)
    listLens = []
    matchPos = []
    codeLens = []
    filtered = []
    totalComps = 0
    for i in range(n):
        l = len(dictionary[phraseGrams[i]])
        listLens.append(l)
        codeLens.append(len(phraseGrams[i]))
    indexes = np.argsort(listLens)
    rarestTerm = phraseGrams[indexes[0]]
     
    candidates = list.copy(dictionary[rarestTerm])
    for i in range(1, n):  
        currentTerm = phraseGrams[indexes[i]]
        delta = indexes[0] - indexes[i]
        
        filtered = []
        
        if delta > 0:
            gap = -np.sum(codeLens[indexes[i]+1:indexes[0]+1])
        else:
            gap = np.sum(codeLens[indexes[0]+1:indexes[i]+1])
            
        for j in range(len(candidates)):
            

            val = candidates[j] - gap       
#            x, searchComps = binarySearchNext(dictionary[currentTerm],
#                                              0, len(dictionary[currentTerm]),
#                                              val)
            searchComps = np.ceil(np.log2(listLens[indexes[i]]))
            totalComps += searchComps
#            if (dictionary[currentTerm][x] == val):
#                filtered.append(candidates[j])
            if (source[val: val + codeLens[indexes[i]]] == currentTerm):
                filtered.append(candidates[j])
            totalComps += 1
        
        candidates = list.copy(filtered)

    matchPos = np.array(filtered) - (np.sum(codeLens) - np.sum(codeLens[0:indexes[0]+1]))
        
    return matchPos, totalComps

def queryIndexFwdOVL(phraseGrams, dictionary, source):
    
    n = len(phraseGrams)
    k = len(phraseGrams[0])
    listLens = []
    matchPos = []
    totalComps = 0
    for i in range(n):
        l = len(dictionary[phraseGrams[i]])
        listLens.append(l)
        
    indexes = np.argsort(listLens)
    rarestTerm = phraseGrams[indexes[0]]
    delta = indexes[0]
            
    candidates = list.copy(dictionary[rarestTerm])
    for i in range(1, n):  
        currentTerm = phraseGrams[indexes[i]]
        delta = indexes[0] - indexes[i]
        
        filtered = []

        for j in range(len(candidates)):
    
            val = candidates[j] - delta
#            x, searchComps = binarySearchNext(dictionary[currentTerm],
#                                              0, len(dictionary[currentTerm]),
#                                              val)
            searchComps = np.ceil(np.log2(listLens[indexes[i]]))
            totalComps += searchComps
#            if (dictionary[currentTerm][x] == val):
#                filtered.append(candidates[j])
            if (source[val: val + k] == currentTerm):
                filtered.append(candidates[j])
            totalComps += 1
        
        candidates = list.copy(filtered)

    matchPos = np.array(filtered) - indexes[0]
                           
    return matchPos, totalComps



##np.random.seed(123)
#N = int(1e7) # sequence length
k = 12 # k length subsequences
qlmin = 50
qlmax = 300
numQueries = 1000
pvals = np.arange(0.5,1,0.1)

with open ('max_len_7_code_books', 'rb') as fp:
    allCodeBooks = pickle.load(fp)

allCodeBooks2 = []
for cb in allCodeBooks:
    A = set(cb)
    ApSet = set(product(A, repeat = 2))
    Ap = [''.join(x for x in code) for code in list(ApSet)]
    allCodeBooks2.append(Ap)
    
avgCBLens = []   
for cb in allCodeBooks2:
    lenSum = 0
    for i in range(len(cb)):
        lenSum += len(cb[i])
    lenSum /= len(cb)
    avgCBLens.append(lenSum) 


with open ('./sources_and_indexes/allSources_1000000', 'rb') as fp:
    X = pickle.load(fp) 
    
with open ('./sources_and_indexes/{}-GramDictionary'.format(k), 'rb') as fp:
    KGramDictionary = pickle.load(fp) 

    
N = len(X[0])
meanComplexityKGrams = []
meanComplexityFFCode = []

for ii in range(105, len(allCodeBooks2)):
    
    codeBook7_i = allCodeBooks2[ii]
        
    complexityKGrams = np.zeros((numQueries, len(pvals)))
    complexityFFCode = np.zeros((numQueries, len(pvals)))
    matchPosKGrams = np.zeros((numQueries, len(pvals)))
    matchPosFFCode = np.zeros((numQueries, len(pvals)))
    
    with open ('./sources_and_indexes/FFdictionary_max_len_14_{}'.format(ii), 'rb') as fp:
        FFdictionary = pickle.load(fp) 
   
    queries = []
    
    queryLengths = np.random.randint(qlmin, qlmax, numQueries)
    queryPos = np.random.randint(0, N - qlmax, numQueries)
    
    for j1 in range(len(pvals)):
        
        codeBook = codeBook7_i
        
        p = pvals[j1]
        Xs = X[j1]
        dictionary = FFdictionary[j1]
        
        phrases = []    
        for i in range(numQueries):
            phrases.append(Xs[queryPos[i]:queryPos[i] + queryLengths[i]])
        
        queries.append(phrases)
        phrases = queries[j1]
        
        for i1 in range(len(phrases)):
            
            phrase = phrases[i1]    
        
            m = len(phrase)
            phraseGramsFwd = []
            phraseGramsBwd = []
#            tail = ''
            
#            phraseFwd = ''
            upper = m-1
            i = 0
            while i < upper:
                j = 1
                while (i+j <= m and not phrase[i:i+j] in codeBook):
                    j = j+1
                if i+j <= m and i == upper-1:
                    code = phrase[i:i+j]
                    phraseGramsFwd.append(code)
#                    phraseFwd += code
#                    tail = phrase[i+j:]
                    i = i+j
        #            print("Forward tail:{}".format(tail))
                    
                elif i+j <= m:
                    code = phrase[i:i+j]
                    phraseGramsFwd.append(code)
#                    phraseFwd += code
                    i = i+j
                else:
    #                phraseFwd += code[1:]
#                    tail = phrase[i+len(code):]
        #            print("Forward tail:{}".format(tail))
                    break
            
#            phraseBwd = ''
            lower = 1
            fwdBwdDelta = 0
            i = m-1
    #        for i in reversed(range(lower , m)):
            while i >= lower:
                j = 0
                while (i-j >= 0 and not phrase[i-j:i+1] in codeBook):
                    j = j+1
                if i-j >= 0 and i == lower:
                    code = phrase[i-j:i+1]
                    phraseGramsBwd.append(code)
#                    phraseBwd += code[::-1]
                    fwdBwdDelta = (i-j)
    #                print(fwdBwdDelta)
                    tail = phrase[:i-j]
                    i = i-j-1
        #            print("Backward tail:{}".format(tail))
                    
                elif i-j >= 0:
                    code = phrase[i-j:i+1]
#                    phraseBwd += code[::-1]
                    phraseGramsBwd.append(code)
                    fwdBwdDelta = (i-j)
    #                print(fwdBwdDelta)
                    i = i-j-1
    
        
                else:

                    break

            totalComps = 0
            
            matchPosFwd, compsFwd = queryIndexFwd(phraseGramsFwd, dictionary, Xs)
            matchPosBwd, compsBwd = queryIndexBwd(phraseGramsBwd, dictionary, Xs)
        
            totalComps = compsFwd + compsBwd
            
        #    print('Total comparisons using the index = %d'%(totalComps))
        
            matchPos = []
            if len(matchPosFwd) > 0 and len(matchPosBwd) > 0:
                for i in range(len(matchPosFwd)):
                    val = matchPosFwd[i] + fwdBwdDelta
                    x, searchComps = binarySearchNext(matchPosBwd, 0, len(matchPosBwd), val)
                    totalComps += searchComps
                    if (matchPosBwd[x] == val):
                        matchPos.append(matchPosFwd[i])
            else:
                totalComps += 2
            
            matchPosFFCode[i1][j1] = len(matchPos)
                
            complexityFFCode[i1][j1] = totalComps
            
#            matchPosPF = []
#            pos = 0
#            while pos >= 0:
#                pos = Xs.find(phrase, pos+1)
#                if pos > 0:
#                    matchPosPF.append(pos)
#                    
#            if not np.all(matchPosPF == matchPos):
#                print('\n***mismatch ff-code {}***\n'.format(ii))
        
        
    codeBook = []
    
    # init postings dictionary    
    for i in range(2**k):
        binVal = toBin(i,k)
        code = ''.join(str(e) for e in binVal)
        codeBook.append(code)
    
    
    for j1 in range(len(pvals)):
        
        p = pvals[j1]
        Xs = X[j1]
        dictionary = KGramDictionary[j1]
            
        phrases = queries[j1]
        
        for i1 in range(len(phrases)):
            
            phrase = phrases[i1]
                
            m = len(phrase)
            phraseGramsFwd = []
            phraseGramsBwd = []
#            tail = ''
            
#            phraseFwd = ''
            upper = m-1
            for i in range(upper):
                j = 1
                while (i+j <= m and not phrase[i:i+j] in codeBook):
                    j = j+1
                if i+j <= m and i == upper-1:
                    code = phrase[i:i+j]
                    phraseGramsFwd.append(code)
#                    phraseFwd += code
#                    tail = phrase[i+j:]
        #            print("Forward tail:{}".format(tail))
                    
                elif i+j <= m:
                    code = phrase[i:i+j]
                    phraseGramsFwd.append(code)
#                    phraseFwd += phrase[i]
                else:
#                    phraseFwd += code[1:]
#                    tail = phrase[i+len(code):]
        #            print("Forward tail:{}".format(tail))
                    break
            
            
            matchPosFwd, compsFwd = queryIndexFwdOVL(phraseGramsFwd, dictionary, Xs)
    #        matchPosBwd, compsBwd = queryIndexBwd(phraseGramsBwd, dictionary)
        
    #        totalComps = compsFwd + compsBwd
            totalComps = compsFwd
            
            matchPos = matchPosFwd
            matchPosKGrams[i1][j1] = len(matchPos)        
                
            complexityKGrams[i1][j1] = totalComps
            
#            matchPosPF = []
#            pos = 0
#            while pos >= 0:
#                pos = Xs.find(phrase, pos+1)
#                if pos > 0:
#                    matchPosPF.append(pos)
#                    
#            if not np.all(matchPosPF == matchPos):
#                print('\n***mismatch k-grams***\n')
        
    
    meanComplexityKGrams_i = np.mean(complexityKGrams, axis = 0)
    meanComplexityFFCode_i = np.mean(complexityFFCode, axis = 0) 
    
    meanComplexityKGrams.append(meanComplexityKGrams_i)
    meanComplexityFFCode.append(meanComplexityFFCode_i)

complexityRatio = np.array(meanComplexityKGrams)/np.array(meanComplexityFFCode)
best_index = np.argmax(np.sum(complexityRatio, axis = 1))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure()
plt.plot(pvals, meanComplexityFFCode[best_index]/1e3, '-s', markersize=8, label = 'FB-SvS with non-overlapping FF code {}'.format(best_index))
plt.plot(pvals, meanComplexityKGrams[best_index]/1e3, '-P', markersize=8, label = 'SvS with overlapping {}-grams code'.format(k))
plt.xlabel('$P_{X}(1)$', fontsize = 16)
plt.ylabel(r'Mean cost $\times 10^3$', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid()
plt.legend(loc = 0, fontsize = 13, frameon = False)
#plt.savefig('FB-SVS_cost_KG_FF_' + str(qlmin) + '_' + str(qlmax) + '_' + str(numQueries) + '.png', dpi = 600, bbox_inches = 'tight')

#with open('./sources_and_indexes/mean_complexity_FFCode_max_len_14_{}_{}_{}_1e6_part4'.format(qlmin, qlmax, numQueries), 'wb') as fp:
#    pickle.dump(meanComplexityFFCode, fp)

#with open('./sources_and_indexes/mean_complexity_12Grams_{}_{}_{}_1e6_part4'.format(qlmin, qlmax, numQueries), 'wb') as fp:
#    pickle.dump(meanComplexityKGrams, fp)



