#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:10:54 2019

@author: jacob
"""
from math import sqrt
from scipy import sparse, optimize
from setToHilbU import *
from scipy.sparse.linalg import expm, expm_multiply
from itertools import combinations 
import numpy as np

##State Repo just makes the relevant states to be used for the cost functions



def spkron(mat0,*mats):
    matkron=mat0
    for mat in mats:
        matkron=sparse.kron(matkron,mat)
    return matkron
def GHZ(x,y,z):
    s0=sparse.coo_matrix([[1],[0]])
    s1=sparse.coo_matrix([[0],[1]])
    if x==y and y==z:
        a=1
    else:
        a=0

    ghzS=sparse.kron(sparse.kron(s0,s0),s0)+sparse.kron(sparse.kron(s1,s1),s1)
    paul1=(x==0)*X+(x==1)*Z
    paul2=(x==0)*X+(x==1)*Z
    paul3=(x==0)*X+(x==1)*Z
    ghzO=((-1)**a)*spkron(paul1,paul2,paul3)*ghzS
    return ghzO


#Proper AMEs

a=np.array([[-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,\
-1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1\
,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,\
1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1]]).T
AME6=a/np.linalg.norm(a)
b=np.array([[1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1]]).T
AME5=sparse.csc_matrix(b/np.linalg.norm(b))


def genGraph(V,E):
    numAtoms=len(V)
    
    psiI=setToHilbU(1/sqrt(2)*sparse.coo_matrix([[1],[1]]),[1],numAtoms,1/sqrt(2)*sparse.coo_matrix([[1],[1]])).todense()
    s0=np.array([[1],[0]])
    s1=np.array([[0],[1]])
    s0t=s0.transpose()
    s1t=s1.transpose()    
    cZ0=s0*s0t
    cZ1=s1*s1t
    I=np.eye(2)
    ys=psiI
    Uabs=[]
    for edgeId in range(0,len(E)):
        edge=list(E[edgeId])
        Uab=setToHilbU([cZ0,cZ0],edge,numAtoms,I).todense()\
        +setToHilbU([cZ0,cZ1],edge,numAtoms,I).todense()\
        +setToHilbU([cZ1,cZ0],edge,numAtoms,I).todense()\
        -setToHilbU([cZ1,cZ1],edge,numAtoms,I).todense()
        #Uabs.append(Uab)
        ys=Uab*ys
    return ys

V=[1,2,3,4,5,6]
E=list(combinations(V,2))
clusterState6=genGraph(V,E)
V=[1,2,3,4]
E=list(combinations(V,2))
clusterState4=genGraph(V,E)
V=[1,2,3,4,5]
E=list(combinations(V,2))
clusterState5=genGraph(V,E)
def dispFunc(f,x):
    a=x
    if len(a)==1:
        a=a.tolist()[0]
    #print(a)
    b=np.array(f(a))
    
    #print(a,'->',b)
    print ("\033[0;0m {} -> \033[1;31m {}".format(a,b))
    #print('probability',b)
    return b
def genControlX(ops,opAtoms,atoms,atomValues,numAtoms):
    #Generates a controlled version of the op gate, where atomvalues are the control,
    #atoms are the actual atoms used in the gate
    if not(isinstance(ops,list)):
        ops=[ops]
    s0=np.array([[1],[0]])
    s1=np.array([[0],[1]])
    s0t=s0.transpose()
    s1t=s1.transpose()    
    Z0=s0*s0t
    Z1=s1*s1t
    I=np.eye(ops[0].shape[0])
    atomMats=[]
    for i in atomValues:
        if i==0:
            atomMats.append(Z0)
        if i==1:
            atomMats.append(Z1)
    proGate=setToHilbU(atomMats+ops,atoms+opAtoms,numAtoms,I)
    iGate=notany(atomMats,atoms,numAtoms)
    return proGate+iGate
    
def notany(matList,atomNums,numAtoms):
    I=np.eye(matList[0].shape[0])
    totU=0*setToHilbU(matList,atomNums,numAtoms,I)
    curMat=[0]*len(matList)
    curMat[-1]=matList[-1]
    state=[0]*len(matList)
    for i in range(1,2**(len(matList))):
        curi=i
        for j in range(len(matList)):
            state[j]=curi % 2
            curi=int(curi/2)
            
            if state[j]==1:
                curMat[j]=I-matList[j]
            else:
                curMat[j]=matList[j]
        totU=totU+setToHilbU(curMat,atomNums,numAtoms,I)
    return totU

def genSwap(i,j,numAtoms):
    #generates an arbitary swap g6ate between atoms i and j, 
    s0=np.array([[1],[0]])
    s1=np.array([[0],[1]])
    s0t=s0.transpose()
    s1t=s1.transpose()
    cZ00=s0*s0t
    cZ01=s0*s1t
    cZ10=s1*s0t
    cZ11=s1*s1t
    Uij=setToHilbU([cZ00,cZ00],[i,j],numAtoms,I)+setToHilbU([cZ11,cZ11],[i,j],numAtoms,I)\
    +setToHilbU([cZ10,cZ01],[i,j],numAtoms,I)+setToHilbU([cZ01,cZ10],[i,j],numAtoms,I)
    return Uij
def printmat(mat):
    if sparse.issparse(mat):
        mat=mat.todense()
    print(np.array2string(mat, threshold=np.inf, max_line_width=np.inf))


#AME Error correction
def genErrGate():
    #Actual local Gates required
    numAtoms=5
    I=np.eye(2)
    cZ=np.array([[1,0],[0,-1]])
    Had=1/sqrt(2)*sparse.csc_matrix([[1,1],[1,-1]])
    cNot=np.array([[0,1],[1,0]])
    
    G1=setToHilbU([Had]*3,[1,2,4],numAtoms,I)
    G2=genControlX(cZ,[5],[2,3,4],[1,1,1],numAtoms)
    G3=genControlX(cZ,[5],[2,3,4],[0,1,0],numAtoms)
    G4=genControlX(cNot,[5],[3],[1],numAtoms)
    G5=genControlX([cNot,cNot],[3,5],[1],[1],numAtoms)
    G6=genControlX([cNot],[3],[4],[1],numAtoms)
    G7=genControlX([cNot],[5],[2],[1],numAtoms)
    G8=genControlX([cZ],[3],[4,5],[1,1],numAtoms)
    return G8*G7*G6*G5*G4*G3*G2*G1
errorgate=sparse.lil_matrix(genErrGate())

