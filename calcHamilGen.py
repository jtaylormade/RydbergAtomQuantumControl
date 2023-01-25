#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:07:48 2019

@author: jacob
"""


#Generic Imports
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
from scipy import sparse as sparse
from scipy.sparse.linalg import LinearOperator
import math
import pickle
#My Imports
from setToHilbU import *

class HamilProp:
  Detuning = None
  numAtoms = 1
  Is = None # list cannot be initialized here!
  posList=None
  calcVijs=None
  NStates=None
  storedTermV=None
  specialCoupling=None
  def __init__(this):
        this.Detuning = None
        this.numAtoms = 1
        this.Is = None # list cannot be initialized here!
        this.posList=None
        this.calcVijs=None
        this.NStates=None
        this.storedTermV=None
        this.specialCoupling=None


def calcHamilGen(hamilConds,t):
    '''
    Calculates the hamiltonian of the multistate, multibody system, at a time t
    This requires a set of properties in the form of hamilConds
    This includes:
    Detuning: List of Detuning functions for each Rydberg level
    numAtoms: the number of bodies in the multibody system
    Is: list of Rabi Frequency functions for different positions and times
    posList: positions of the many bodies
    NStates: the number of states including the ground to simulate
    
    '''
    if not(isinstance(hamilConds.Detuning,list)):
        hamilConds.Detuning=[hamilConds.Detuning]
    if not(isinstance(hamilConds.Is,list)):
        hamilConds.Is=[hamilConds.Is]
    if not(isinstance(hamilConds.calcVijs,list)):
        hamilConds.calcVijs=[hamilConds.calcVijs]
    Detuning=[0]*len(hamilConds.Detuning)
    numAtoms=hamilConds.numAtoms
    Is=hamilConds.Is
    posList=hamilConds.posList
    calcVijs=hamilConds.calcVijs
    NStates=hamilConds.NStates
    specialCoupling=hamilConds.specialCoupling
    ct=lambda mat: mat.getH()
    setToHilb=lambda mat, atom: setToHilbU(mat,atom,numAtoms,sparse.eye(NStates))
    zeroTerm=setToHilb(sparse.eye(NStates)*0,1)
    termA=zeroTerm
    termV=zeroTerm
    Sg=sparse.eye(1,NStates,0).T
    Sr=[0]*(NStates-1)
    #Insert Something about only calculating TermV once
    for k in range(0,NStates-1):
        if callable(hamilConds.Detuning[k]):
            Detuning[k]=hamilConds.Detuning[k](t)
        else:
            Detuning[k]=hamilConds.Detuning[k]
        Sr[k]=sparse.eye(1,NStates,k+1).T
    #Calculate the non-interacting parts of the hamiltonian
    def termAs(index):
        i=math.floor(index/(NStates-1))
        k=index % (NStates-1)
        tTerm1=-Detuning[k]*Sr[k]*ct(Sr[k])
        if callable(Is[k]):
            rabi=Is[k](posList[i],t)/2
        else:
            rabi=Is[k]/2
        tTerm2=rabi*Sr[k]*ct(Sg)
        tTerm2=tTerm2+ct(tTerm2)
        tTermA=setToHilb(tTerm1+tTerm2,i+1)
        return tTermA

    for index in range(0,numAtoms*(NStates-1)):
        termA=termA+termAs(index)
    if specialCoupling is not None:
        for i in range(numAtoms):
            for spC in specialCoupling:
                k0=spC[0][0]-1
                k1=spC[0][1]-1
                if k0 ==-1:
                    Sa=Sg
                else:
                    Sa=Sr[k0]
                if k1 ==-1:
                    Sb=Sg
                else:
                    Sb=Sr[k1]
                if callable(spC[1]):
                    rabi=spC[1](posList[i],t)/2
                else:
                    rabi=spC[1]/2
                tTerm2=rabi*Sb*ct(Sa)
                tTerm2=tTerm2+ct(tTerm2)
                tTermA=setToHilb(tTerm2,i+1)
                termA=termA+tTermA
    def termVs(index):
        #Converts single index into 
        tindex=index
        i=tindex % numAtoms
        tindex=math.floor(tindex/numAtoms)
        k=tindex % (NStates-1)
        tindex=math.floor(tindex/(NStates-1))
        j=tindex
        if j>i:
            doubleExcited=[Sr[k]*ct(Sr[k])]*2
            atomIndex=[i+1,j+1]
            relHilb=setToHilb(doubleExcited,atomIndex)
            if callable(calcVijs[k]):
                ttermV=calcVijs[k](posList[i],posList[j])*relHilb
            else:
                ttermV=calcVijs[k]*relHilb
        else:
            ttermV=zeroTerm
        return ttermV
    
    if hamilConds.storedTermV is None:
        calcTermV=1
    else:
        termV=hamilConds.storedTermV
        calcTermV=0
    if calcTermV:
        for index in range(0,(numAtoms**2)*(NStates-1)):
            termV=termV+termVs(index)
        hamilConds.storedTermV=termV
    return termA + termV