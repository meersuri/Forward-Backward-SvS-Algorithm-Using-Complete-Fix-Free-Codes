# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:38:11 2019

@author: Meer Suri
"""

import itertools
#from itertools import product, combinations
import numpy as np
import scipy
from scipy.special import binom 
import pickle


class Node:
    parent = None
    def __init__(self, data, parent_node = None):
        self.data = data
        self.left = None
        self.right = None
        if not parent_node == None:
            self.parent = parent_node
    def extendLeft(self):
        self.left = Node('0', self)
    def extendRight(self):
        self.right = Node('1', self)
        
class NodeN:
    parent = None
    def __init__(self, data, parent_node = None):
        self.data = data
        self.children = []
        self.pruned = False
        if not parent_node == None:
            self.parent = parent_node
    def addChild(self):
        self.children.append(NodeN(len(self.children), self))
    
    def prune(self):
        self.pruned = True
        
               
#def toBin(n,width):
#    x = bin(n)[2:]
#    x = x.zfill(width)
#    l = len(x)
#    b = np.zeros(l,dtype='int32')
#    for i in range(l):
#        if x[i] == '1':
#            b[i] = 1
#        else:
#            b[i] = 0
#    return b

def to4ary(n, width):
    x = n
    FaryStrRev = ''
    while x > 0:
        rem = x - (x//4)*4
        FaryStrRev  += str(rem)
        x = x//4
    FaryStr = FaryStrRev[::-1]
    if len(FaryStr) < width:
        n = width - len(FaryStr)
        FaryStr = '0'*n + FaryStr
    return FaryStr


def calcDegree(source):
    weights = np.zeros(len(source))
    for i in range(len(source)):
        weights[i] = (i+1)*source[i]*4**-(i+1)
    degree = np.sum(weights)
    return degree

def calcKraftSum(source):
    weights = np.zeros(len(source))
    for i in range(len(source)):
        weights[i] = source[i]*4**-(i+1)
    kraftSum = np.sum(weights)
    return kraftSum

def getLeafNodes(root):
    nodesAtCurrentLevel = [root]
    leafNodes = []
    while True:
        nodesInNextLevel = []
        for node in nodesAtCurrentLevel:
            if len(node.children) == 0:
                leafNodes.append(node)
            else:
                nodesInNextLevel.extend(node.children)
        if len(nodesInNextLevel) == 0:
            break
        else:
            nodesAtCurrentLevel = list.copy(nodesInNextLevel)
    
    return leafNodes

def getLeafNodesSubTree(rootSubTree):
    leafNodesSubTree = []
    nodesAtCurrentLevelSubTree = [rootSubTree]
    while True:
        nodesInNextLevelSubTree = []
        for nodeSubTree in nodesAtCurrentLevelSubTree:
            if nodeSubTree.left == None and nodeSubTree.right == None:
                leafNodesSubTree.append(nodeSubTree)
            else:
                nodesInNextLevelSubTree.append(nodeSubTree.left)
                nodesInNextLevelSubTree.append(nodeSubTree.right)
        if len(nodesInNextLevelSubTree) == 0:
            break
        else:
            nodesAtCurrentLevelSubTree = list.copy(nodesInNextLevelSubTree)
        
    return leafNodesSubTree
    

def getCodeBook(leafNodes):
    codeBook = []
    revCodeBook = []
    codeLens = []

    for leaf in leafNodes:
        code = [leaf.data]
        parent = leaf.parent
        while parent != None:
            code.append(parent.data)
            parent = parent.parent
        code = code[:-1]
        code.reverse()
        codeStr = ''.join(str(e) for e in code)
        codeBook.append(codeStr)
        revCodeBook.append(codeStr[::-1])
        codeLens.append(len(codeStr))
    
    sortedCodeBook = sorted(revCodeBook)
    fixFree = True
    for i in range(len(sortedCodeBook) - 1):
        currentCode = sortedCodeBook[i]
        nextCode = sortedCodeBook[i + 1]
        if nextCode[:len(currentCode)] == currentCode:
            fixFree = False
            break
    
    return fixFree, codeBook, 

#def checkComplementCode(codeBook, nm):
#    count = 0
#    for code in codeBook:
#        if code[0] == '0':
#            count += 1
#    
#    if count >= np.ceil(nm/2):
#        return True
#    else:
#        return False


def getCodeStr(leaf):
    code = leaf.data
    parent = leaf.parent
    while parent != None:
        code = parent.data + code
        parent = parent.parent
    codeStr = code[2:]
    return codeStr

def checkSuffixFree(code, sortedCodeBook, runningCodeBookLen):
    suffixFree = True
    for i in range(runningCodeBookLen):
        currentCode = sortedCodeBook[i]
#        nextCode = sortedCodeBook[i + 1]
        if code[-1 - (len(list(currentCode))-1):] == currentCode:
            suffixFree = False
            break
    return suffixFree

def checkShiftRegSeq(codeBook):
    shiftRegSeqFree = True
    successor = {}
    predecessor = {}
    edges = []
    noIncomingNodes = set(codeBook)
    for code in codeBook:
        successor[code] = []
        for code2 in codeBook:
            if code == code2:
                continue
            elif code2[:-1] == code[1:]:
                successor[code].append(code2)
                edges.append((code,code2))
                noIncomingNodes.discard(code2)
                
    for code in codeBook:
        predecessor[code] = []
        for code2 in codeBook:
            if code == code2:
                continue
            elif code2[1:] == code[:-1]:
                predecessor[code].append(code2)
    
    L = []
    S = list(noIncomingNodes)
    while len(S) > 0:
        node = S.pop(len(S)-1)
        L.append(node)
        nodeSuccessors = successor[node]
        for succ in nodeSuccessors:
            edges.remove((node,succ))
            predecessor[succ].remove(node)
            if len(predecessor[succ]) == 0:
                S.append(succ)
                
    if len(edges) > 0:
        shiftRegSeqFree = False
    
    return shiftRegSeqFree
    

allSources = []
completeSources = []
candidateSources = []
completeSourceDegrees = []
allFixFreeCodes = []
allCodeBooks = []

root = NodeN(-1)
superTree = [root]
level = 0
maxLevel = 4 

blockSources = []
for i in range(maxLevel):
    source = [0]*maxLevel
    source[i] = 4**(i+1)
    blockSources.append(source)
    
    
nodesAtCurrentLevel = [root]
with open('./sources_tree_data/level_-1_node_0_children'.format(level,i), 'wb') as fp:
    pickle.dump([root], fp)
    
numLeaves = 1

cval = 1
while level < maxLevel:
    counter = 0
#    nodesInNextLevel = []
    for i1 in range(cval):
        
        with open('./sources_tree_data/level_{}_node_{}_children'.format(level-1, i1), 'rb') as fp:
            nodesAtCurrentLevel = pickle.load(fp)
        
            
        for i in range(len(nodesAtCurrentLevel)):
            currentNodeChildren = []
            prevLevelsNodeCounts = [nodesAtCurrentLevel[i].data]
            parent = nodesAtCurrentLevel[i].parent
            while parent != None:
                prevLevelsNodeCounts.append(parent.data)
                parent = parent.parent
            del prevLevelsNodeCounts[-1]
            prevLevelsNodeCounts = np.array(prevLevelsNodeCounts)
            usedNodes = np.zeros(len(prevLevelsNodeCounts))
            
            for j in range(len(usedNodes)):
                usedNodes[j] = prevLevelsNodeCounts[j]*4**(j+1)
                
            maxNodes = 4**(level+1)+1 - np.sum(usedNodes)
            
            for j in range(int(maxNodes)):
                nodesAtCurrentLevel[i].addChild()
    #            nodesInNextLevel.append(nodesAtCurrentLevel[i].children[-1])
                currentNodeChildren.append(nodesAtCurrentLevel[i].children[-1])
            
            with open('./sources_tree_data/level_{}_node_{}_children'.format(level, counter), 'wb') as fp:
                pickle.dump(currentNodeChildren, fp)
                
            counter += 1
    
    print(counter)
    cval = counter
    numLeaves = len(nodesAtCurrentLevel)
#    nodesAtCurrentLevel = list.copy(nodesInNextLevel)
    level += 1
    
#print(cval)
leafNodes = []
for i in range(cval):
     with open('./sources_tree_data/level_{}_node_{}_children'.format(maxLevel-1, i), 'rb') as fp:
         nodes = pickle.load(fp)
     leafNodes.extend(nodes)
 
#leafNodes = getLeafNodes(root)
       
for leaf in leafNodes:
    source = [leaf.data]
    parent = leaf.parent
    while parent != None:
        source.insert(0, parent.data)
        parent = parent.parent
    del source[0]
    allSources.append(source)
#    if calcKraftSum(source) == 1.0:
#        completeSources.append(source)
    

for source in allSources:
    if calcKraftSum(source) == 1.0:
        completeSources.append(source)
    
for source in completeSources:
    completeSourceDegrees.append(calcDegree(source))

for i in range(len(completeSources)):
    source = completeSources[i]
    degree = completeSourceDegrees[i]
    if int(degree) == degree and source not in blockSources:
        candidateSources.append(source)

allRunningCodeBooks = []
allSubTreeCodeBooks = []
for source in candidateSources:
    
    availNodes = [0]*len(source)
    combinations = [0]*len(source)
    for i in range(len(source)):
        usedNodes = 0
        for j in range(i):
            usedNodes += source[j]*4**(i - j)
        availNodes[i] = 4**(i+1) - usedNodes
        combinations[i] = int(scipy.special.binom(availNodes[i], source[i]))
        
    print("avail  :{}".format(availNodes))    
    print("source :{}".format(source))
    print("combinations :{}".format(combinations))    
    
    root = NodeN(-1)
    level = 0
    maxLevel = len(source)
    nodesAtCurrentLevel = [root]

    superTreeChildren = []
#    allAllAvailNodesSubTrees = []
#    allLevelNodesInSuperTree = []
    while level < maxLevel:
        nodesInNextLevel = []
#        allAvailNodesSubTrees = []
#        superNodeFixFreeNodes = []
#        superNodeSubTrees = []
#        allLevelNodesInSuperTree.append(len(nodesAtCurrentLevel))
        for nodeSuperTree in nodesAtCurrentLevel:
            #check if sub-tree rooted at this node is fix-free
            subTree = [nodeSuperTree.data]
            parent = nodeSuperTree.parent
            while parent != None:
                subTree.insert(0, parent.data)
                parent = parent.parent
            del subTree[0]
#            superNodeSubTrees.append(subTree)            

            rootSubTree = NodeN('-1')
            levelSubTree = 0
            maxLevelSubTree = level
            nodesAtCurrentLevelSubTree = [rootSubTree]
            runningLeafNodes = []
            runningCodeBook = []
            runningCodeBookLen = 0
            availNodesSubTree = list.copy(availNodes)
            availNeutralNodes = []

            while levelSubTree <= maxLevelSubTree:
                nodesInNextLevelSubTree = []
                neutralNodesCurrentLevel = []
                for nodeSubTree in nodesAtCurrentLevelSubTree:
                    for i in range(4):
                        nodeSubTree.addChild()
                        node = nodeSubTree.children[-1]
                        nodeCode = getCodeStr(node)
                        nodeSuffixFree = checkSuffixFree(nodeCode, runningCodeBook, runningCodeBookLen)
                        if not nodeSuffixFree:
                            nodesInNextLevelSubTree.append(nodeSubTree.children[-1])
                        else:
                            neutralNodesCurrentLevel.append(nodeSubTree.children[-1])
                
                internalNodes = len(nodesInNextLevelSubTree)
                neutralNodes = len(neutralNodesCurrentLevel)
                availNeutralNodes.append(neutralNodes)
                numChildren = int(scipy.special.binom(neutralNodes, source[len(availNeutralNodes)-1]))
                
                if numChildren > 1e2:
                    nodeSuperTree.prune()

                if levelSubTree < maxLevelSubTree:
                    
                    if source[len(availNeutralNodes)-1] > 0 and neutralNodes >= source[len(availNeutralNodes)-1]:
                        allCombinations = list(itertools.combinations(range(neutralNodes), source[len(availNeutralNodes)-1]))
                        leafNodesAtNextLevelSubTree = allCombinations[subTree[len(availNeutralNodes)-1]]
    #                   
                        for i1 in list(leafNodesAtNextLevelSubTree)[::-1]:
                            runningLeafNodes.append(neutralNodesCurrentLevel.pop(i1))
                            nodeCode = getCodeStr(runningLeafNodes[-1])
                            suffixFree = checkSuffixFree(nodeCode, runningCodeBook, runningCodeBookLen)
                            if not suffixFree:
                                nodeSuperTree.prune()
                                break
                            else:
                                runningCodeBook.append(getCodeStr(runningLeafNodes[-1]))
                                runningCodeBookLen += 1
                        
                        if level == np.nonzero(source)[0][0]+1 and not nodeSuperTree.pruned:
#                            print('level {} pruning'.format(levelSubTree))
                            allSubTreeCodeBooks.append(runningCodeBook[:source[levelSubTree]])
#                            complementCode = checkComplementCode(runningCodeBook[:source[levelSubTree]], source[levelSubTree])
#                            if not complementCode:
#                                nodeSuperTree.prune()
                            shiftRegSeqFree = checkShiftRegSeq(runningCodeBook[:source[levelSubTree]])
                            if not shiftRegSeqFree:
                                nodeSuperTree.prune()
#                                print('pruned shift reg seq')
                     
                    elif neutralNodes < source[len(availNeutralNodes)-1]:
                        nodeSuperTree.prune()
#                        print('insufficient nodes')
#                    break

                   
                else:
                    
                    if neutralNodes < source[len(availNeutralNodes)-1]:
                        nodeSuperTree.prune()
                    else:
                        for i1 in reversed(range(len(neutralNodesCurrentLevel))):
                            runningLeafNodes.append(neutralNodesCurrentLevel.pop(i1))
                            nodeCode = getCodeStr(runningLeafNodes[-1])
                            suffixFree = checkSuffixFree(nodeCode, runningCodeBook, runningCodeBookLen)
                            if not suffixFree:
                                nodeSuperTree.prune()
                                break
                            else:
                                runningCodeBook.append(getCodeStr(runningLeafNodes[-1]))
                                runningCodeBookLen += 1
                            if not nodeSuperTree.pruned and len(runningCodeBook) == sum(source):
                                allRunningCodeBooks.append(runningCodeBook)
                    
                nodesInNextLevelSubTree.extend(neutralNodesCurrentLevel)
            
                nodesAtCurrentLevelSubTree = list.copy(nodesInNextLevelSubTree)
                levelSubTree += 1
                               
            if sum(source[:len(subTree)]) == 0:
                numChildren = combinations[level]

#            print(availNeutralNodes)
#            print('level {}'.format(level))
###            print('subTree level {}'.format(levelSubTree-1))
#            print('children = {}'.format(numChildren))
#            print('subTree = {}'.format(subTree))
            
           
            if not nodeSuperTree.pruned:
                for i in range(numChildren):
                    nodeSuperTree.addChild()
                    nodesInNextLevel.append(nodeSuperTree.children[-1])
        
        superTreeChildren.append(numChildren)
#        print(len(nodesInNextLevel))
        print(numChildren)
        nodesAtCurrentLevel = list.copy(nodesInNextLevel)
        
        if len(nodesInNextLevel) == 0:
            break
        else:
            level += 1
            
#    allFixFreeCodes.append(allRunningCodeBooks)
    
#    
#with open('max_len_{}_code_books.txt'.format（max, 'w') as f:
#    for codeBook in allRunningCodeBooks:
#        f.write("%s\n" % codeBook)  



#with open('max_len_{}_code_books'.format(maxLevel), 'wb') as fp:
#    pickle.dump(allRunningCodeBooks, fp)
    

    
    
        






