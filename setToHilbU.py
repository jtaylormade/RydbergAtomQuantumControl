#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:04:57 2019

@author: jacob
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix,csc_matrix
from scipy import sparse

def sparseEqual(A,B):
    """
    Checks if two sparse matrices are exactly equal
    @param A: first matrix
    @param B: second matrix
    """
    if A.get_shape()==B.get_shape():
        if (A-B).nnz==0:
            return 1
    return 0

#This Funct
def setToHilbU(matrix,atomNum,totAtomNum,defaultState):
    """
    @param matrix: This is a list of matrices other than defaultState used
    @param atomNum: Positions of non-default Matrices in Tensor Product
    @param totAtomNum: total number of atoms in hilbert space
    @param defaultState: The matrix used when not in atomNum positions
    @return: The matrices and defaultstate tensor producted into the hilbert space
    """
    #Not Required in Matlab but required in Python as int has no len
    if not(isinstance(atomNum, list)):
        atomNum=[atomNum]
    #Repeat Matrix if only one matrix was inputted
    if not(isinstance(matrix, list)):
       matrix=[matrix]*len(atomNum)
    #Sort atomNum and Matrices
    matrix=[x for _,x in sorted(zip(atomNum,matrix))]
    atomNum.sort()
    atomNum=atomNum+[totAtomNum+1]
    curIndex=0
    #SetDefaultMatrix
    defaultState=sparse.csc_matrix(defaultState)
    #Set the starting matrix, since kron(1,mat)=mat this handles 0 endpoint
    tMat=1
    noStepMat=0
    curCount=1
    #No last index
    lastI=0
    #Okay this code is ported from Matlab, It is probably pretty confusing
    #One complication is matlab indexing starts at 1 while pythons starts at 0
    #So there will be a lot instances of i-1 situations
    for i in atomNum:
        #First lets deal with the errors
        if i==lastI:
            continue
        if i<1:
            continue
        #Now since the list of atomNums can go beyond the max # of atoms
        #Set the index to the last atom if i is beyond the last
        if i >totAtomNum:
            i=totAtomNum
            #Don't use the step Matrix at the end
            noStepMat=1
        #The Default matrix will be repeated a different number of times n
        if not(noStepMat):
            stepMat=csc_matrix(matrix[curCount-1])
            n=(i)-(lastI+1)
        else:
            n=(i)-(lastI+1)+1
        
#Determine the tC or matrix to use in this iteration
        if sparseEqual(defaultState,sparse.eye(defaultState.shape[1])) and n!=0:
            if noStepMat:
                #If the step mat is not included generate kron of identities only
                tC=sparse.eye((defaultState.shape[1])**n)
            else:
                #If our matrix is included hthen include it in our kron
                tC=sparse.kron(sparse.eye((defaultState.shape[1])**n),stepMat)
        elif n==0:
            #if there is no default mats included in this step
            tC=stepMat
        else:
            #For non-identity matrices we don't have shortcuts
            tC=defaultState
            if n>1:
                for index in range(1,n):
                    tC=sparse.kron(tC,defaultState)
            if not(noStepMat):
                tC=sparse.kron(tC,stepMat)
#Actually kron it to the current cummulative matrix
        tMat=sparse.kron(tMat,tC)
        lastI=i
        curCount=curCount+1
        if i== totAtomNum:
            break
    return tMat


        

