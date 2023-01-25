#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:40:09 2019

@author: jacob
"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"]="1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
import StateRepo
from StateRepo import printmat
import time
import StateRepo
import math,cmath
from cmath import exp
from pyDOE import lhs
import numpy as np
from numpy import eye
from scipy import sparse, optimize,linalg
from setToHilbU import *
from scipy.sparse.linalg import expm, expm_multiply
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import calcHamilGen
from math import pi,sqrt
import functools
#from yabox import PDE,DE
#import pyswarm
global minima
import logging
#import blackbox as bb
from numpy.random import rand
disfver=''
minima=[]
numAtoms=2
logging.shutdown()
logging.basicConfig(filename=('logs/savedMinima.log'),level=logging.DEBUG)
I=sparse.eye(3)
Z=sparse.csc_matrix([[1,0],[0,-1]])
X=sparse.csc_matrix([[0,1],[1,0]])
wantedZ=expm(-1j*pi*setToHilbU([Z],[1],numAtoms,I).todense())
Vij=300
P=None
interp=None
global zzrabi
global zrabi
global xrabi
zzrabi=20
zrabi=10
def dispFunc(f,x):
  #dispFunc is merely a printing function, it runs f with input x, but prints
  #both the input and output before returning it.
    a=x
    if len(a)==1:
        a=a.tolist()[0]
    #print(a)
    out=f(a)
    if sparse.issparse(out):
      out=out.todense()
    if isinstance(out,float):
      b=out
    else:
      b=np.array(out)
      b=np.asscalar(out)
    
    #print(a,'->',b)
    global disIndex
#    disIndex=disIndex+1
    print ("\033[0;0m {} -> \033[1;31m {}".format(a,b))
        
    #print('probability',b)
    return b
  


def directQAOA(times,psiI,ver=7,parallel=1,op=0):
  #direct implies it does the full hamiltonian version of the pulse sequence
  #This function is the main simulation function, it takes in a series of parameters
  #That have first gone through ParseInput, and then runs the respective hamiltonian and unitary calculations required
  #There exists 8 different version of this simulation and parsing however most are deprecated
  #Only versions 7 and 8 are actually used.
  #Version 7 is the full version that runs without any assumption of symmetry
  #I.e. Hz even and Hz Odd do not have the same time same with Hzz
  #version 8 is the symmetric version, in it they do have the same time
  #Param: times is the parsed times from parseInput, that will be used in the simulation
  #Param: psiI is the initial quantum state of the system
  #Param parallel is whether to run the simualtion in parallel, this is using parallelization for a single run.
  #Param op: is whether or not an operator is being outputed instead
    ys=psiI
    def hamil(indexTime,numAtoms=numAtoms):
        if isinstance(indexTime,tuple):
            index=indexTime[0]
            time=indexTime[1]
        else:
            index=indexTime
            time=times[index]
#        if abs(times[index])<1E-2:
#          times[index]=np.sign(times[index])*1E-2
        if ver==0:
            if index % 3 ==0:    
                #tHz1
                tH=gensZ(times[index],3,numAtoms)
            elif index % 3 ==1:
                #tHzz1
                tH=genZZF(times[index],3,numAtoms)
            elif index % 3 ==2:
                #tHx
                tH=genX(times[index],numAtoms)
        elif ver==1:
            step=index % 5
            if step ==0:    
                #tHz1
                tH=gensZ(times[index],0,numAtoms)
            elif step==1:  
                #tHz2
                tH=gensZ(times[index],1,numAtoms)
            elif step==2:
                #tHzz1
                tH=genZZF(times[index],0,numAtoms)
            elif step ==3:    
                #tHz1
                tH=genZZF(times[index],1,numAtoms)
            elif step==4:
                #tHx
                tH=genX(times[index],numAtoms)
        elif ver==2:
            step=index % 2
            if step ==0:    
                #tHzz
                tH=genZZF(times[index],3,numAtoms)
            elif step==1:  
                #tHx
                tH=genX(times[index],numAtoms)
        elif ver==3:
            step=index % 5
            if step ==0:    
                #tHz1
                tH=gensZt(times[index],0,numAtoms)
            elif step==1:  
                #tHz2
                tH=gensZt(times[index],1,numAtoms)
            elif step==2:
                #tHzz1
                tH=genZZF(times[index],0,numAtoms)
            elif step ==3:    
                #tHz1
                tH=genZZF(times[index],1,numAtoms)
            elif step==4:
                #tHx
                tH=genX(times[index],numAtoms)
        elif ver==4:
            step=index % 4
            if step ==0:    
                #tHz1
                tH=gensZ(times[index],0,numAtoms)
            elif step==1:  
                #tHz2
                tH=gensZ(times[index],1,numAtoms)
            elif step==2:
                #tHzz1
                tH=genZZF(times[index],0,numAtoms)
            elif step ==3:    
                #tHz1
                tH=genZZF(times[index],1,numAtoms)   
        elif ver==5:
            step=index % 5
            if step ==0:    
                #tHz1
                tH=gensZt(times[index],0,numAtoms)
            elif step==1:  
                #tHz2
                tH=gensZt(times[index],1,numAtoms)
            elif step==2:
                #tHzz1
                tH=genZZF(times[index],0,numAtoms,1)
            elif step ==3:    
                #tHz1
                tH=genZZF(times[index],1,numAtoms,1)
            elif step ==4:    
                #tHz1
                tH=genX(times[index],numAtoms)
        elif ver==6:
            step=index % 3
            if step ==0:    
                #tHz1
                tH=gensZt(times[index],1,numAtoms)*gensZt(times[index],0,numAtoms)
            elif step==1:
                #tHzz1
                tH=genZZF(times[index],3,numAtoms,1)
            elif step ==2:    
                #tHz1
                tH=genX(times[index],numAtoms)   
                #tHz1
        elif ver==7:
            step=index % 5
            if step ==0:  
                tH=genX(times[index],numAtoms)  
                
            elif step==1:  
                tH=gensZt(times[index],0,numAtoms)

                
            elif step==2:
                tH=gensZt(times[index],1,numAtoms)

                
            elif step ==3:    

                tH=genZZF(times[index],0,numAtoms,1)
                
            elif step ==4:    

                tH=genZZF(times[index],1,numAtoms,1)
        elif ver==8:
            step=index % 3
            if step ==0:  
                tH=genX(times[index],numAtoms)  
            elif step==1:  
                tH=gensZt(times[index],0,numAtoms)*gensZt(times[index],1,numAtoms)
            elif step ==2:
                tH=genZZF(times[index],0,numAtoms,1)*genZZF(times[index],1,numAtoms,1)
        return tH
    if parallel:
        if __name__ == '__main__':
            thamil=lambda indexTime,numAtoms=numAtoms:hamil(indexTime,numAtoms)
            tHs=list(p.map(thamil,zip(list(range(0,len(times))),times)))
    else:
        tHs=list(map(hamil,range(len(times))))  
    if not(op):
      for tH in tHs:
          ys=tH*ys
    else:
      ys=tHs[0]
      for tH in tHs[1:]:
          ys=tH*ys
    return ys

def parseInput(inputTimes,ver=0):
    #Parses a list of x values into a list of lists that contain the parameters
    #required for each step in time
    if ver==0:
        paramReq=[2,2,1]
    elif ver==1:
        paramReq=[2,2,2,2,1]
    elif ver==2:
        paramReq=[2,1]
    elif ver==3:
        paramReq=[1,1,2,2,1]
    elif ver==4:
        paramReq=[2,2,2,2]
    elif ver==5 or ver==7:
        paramReq=[1,1,1,1,1]
    elif ver==6 or ver ==8:
        paramReq=[1,1,1]
    steps=len(paramReq)
    outputTimes=[]
    i=0
    index=0
    while(index<len(inputTimes)):
        paramStep=i % steps
        if paramReq[paramStep]==1:
            outputTimes.append(inputTimes[index])
            index=index+1
        else:
            stepSize=paramReq[paramStep]
            outputTimes.append(inputTimes[index:index+stepSize])
            index = index+stepSize
        i=i+1
    #print(outputTimes)
    return outputTimes

def genEffState_Dep(state):
    #DEPRECATED
    s=[0]*2
    s2=[0]*2
    s[0]=np.array([[1,0,0]]).T
    s[1]=np.array([[0,1,0]]).T
    s2[0]=np.array([[1,0]]).T
    s2[1]=np.array([[0,1]]).T
    svec=sparse.lil_matrix(0j*sparse.eye(1,2**numAtoms,1).T)
    for i1 in range(2**numAtoms):
        curi1=i1
        for j1 in range(numAtoms):
            index1=curi1%2
            curi1=int(curi1/2)
            if j1==0:
                ss1=s[index1]
                ss2=s2[index1]
            else:
                ss1=sparse.kron(ss1,s[index1])
                ss2=sparse.kron(ss2,s2[index1])
        svec[ss2.nonzero()[0][0]]=np.complex(ss1.T*state)
    return svec
def genProj(numAtoms):
    #Generate a projector that takes a matrix from 3 level numAtoms system to 2 level numAtoms system
    #This gets rydberg of the Rydberg state
    s2=[0]*2
    sr=[0]*2
    s2[0]=np.array([[1,0]]).T
    s2[1]=np.array([[0,1]]).T
    sr[0]=np.array([[1,0,0]]).T
    sr[1]=np.array([[0,1,0]]).T
    Pparts=s2[0]*sr[0].T
    for i in range(1,len(sr)):
        Pparts=Pparts+(s2[i]*sr[i].T)
    Plocals=[Pparts]*numAtoms
    P=spkronl(Plocals)
    return P
def genEffState(state):
    #Convert 3 level state to 2 level state.
    global P
    if P is None or P.shape[1]!=state.shape[0]:
        P=genProj(numAtoms)
    svec=P*state
    return svec
def genEffH(mat,numAtoms=5):
    #given a matrix convert a 3 level system to a 2 level system
    #where numAtoms is the number of atoms.
    global P
    if P is None or P.shape[1]!=mat.shape[0]:
        P=genProj(numAtoms)
    smat=P*mat*P.T

    return smat
def genEffH_dep(mat,numAtoms=2):
    '''Find a faster way to do this operation'''
    #DEPRECATED
    s=[0]*2
    s[0]=np.array([[1,0,0]]).T
    s[1]=np.array([[0,1,0]]).T
    smat=sparse.eye(2**numAtoms)*0j
    smat=sparse.lil_matrix(smat)
    for i1 in range(2**numAtoms):
        curi1=i1
        for j1 in range(numAtoms):
            index1=curi1%2
            curi1=int(curi1/2)
            if j1==0:
                ss1=s[index1]
            else:
                ss1=sparse.kron(ss1,s[index1])  
        for i2 in range(2**numAtoms):
            curi2=i2
            for j2 in range(numAtoms):
                index2=curi2%2
                curi2=int(curi2/2)
                if j2==0:
                    ss2=s[index2]
                else:
                    ss2=sparse.kron(ss2,s[index2]) 
            if sparse.issparse(ss1):
                ss1=ss1.todense()
            comp=np.complex(ss1.T*mat*ss2)
            smat[i1,i2]=comp
    return smat
def genX(time,numAtoms=4):
    #Generates the Hx time evolution step given an effective time.
    time=time#*np.asscalar(rand(1)/100+1)
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[0,0]
    #hamilConds.Is=3
    hamilConds.Is=[lambda y,t:1,0]
    hamilConds.NStates=3
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*1*time*hamil1)
    return apphamil
    #appeff=genEffectHamil(apphamil,numAtoms)
    #return appeff
def genZZ(x,even=0,numAtoms=2):
    #DEPRECATED
    #gensZZ generates the Hzz time evolution instead of from the effective times, directly with an x.
    
    x=np.array(x)
    phase2=abs(x[0])
    Detuning1=x[1]
    t=2*pi/sqrt(x[1]**2+x[0]**2)#*np.asscalar(rand(1)/100+1)
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[0,0]
    #hamilConds.Is=3
    hamilConds.Is=[lambda y,t: 0*(y[0]==0)*(y[1]==0)]*2
    hamilConds.NStates=3
    coupling=[1,2]
    if even:
        hamilConds.specialCoupling=(list(zip([coupling],[lambda y,t: pi*(y[1] % 2==0)])))
    else:
        hamilConds.specialCoupling=(list(zip([coupling],[lambda y,t: pi*(y[1] % 2==1)])))
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[0,Detuning1]
    #hamilConds.Is=3
    hamilConds.Is=[lambda y,t: 0*(y[0]==0)*(y[1]==0)]*2
    hamilConds.NStates=3
    if even:
        hamilConds.specialCoupling=(list(zip([coupling],[lambda y,t: phase2*(y[1] % 2==1)])))
    else:
        hamilConds.specialCoupling=(list(zip([coupling],[lambda y,t: phase2*(y[1] % 2==0)])))
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil2=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*1*hamil1)*expm(-1j*t*hamil2)*expm(-1j*1*hamil1)
    return apphamil
def genZZF(x,odd=0,numAtoms=4,mver=1):
    #genZZF takes in a Detuning and Rabi Frequency x[0] and x[1] and whether to
    #run on odd atoms, even or for odd==3 both
    #For 4 atoms the process is hard coded in, however for 4+ atoms the a general
    #code is used. The general code does not work with 4 atoms or lower but inessence
    #does the same thing as the 4 atom case.do
    global zzrabi
    if mver==0:

        m=genZZ(x[0:2],0)*genZZ(x[0:2],1)
    else:
        tino=(x +math.pi)%(2*math.pi)-math.pi
        if tino==0:
            tino=1E-9
        tin=math.pi-abs(tino)
        rabi=zzrabi
        detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
        m=genZZ([rabi,detuning],0)
    if numAtoms==4:

        if odd==0:
            return spkronl([m,eye(3),eye(3)])*spkronl([eye(3),eye(3),m])
        elif odd==1:
            return spkronl([eye(3),m,eye(3)])
        else:
            return spkronl([eye(3),m,eye(3)])*spkronl([m,eye(3),eye(3)])*spkronl([eye(3),eye(3),m])
    else:
        
        #wantedZZ=expm(-1j*x[0]*setToHilbU([Z,Z],[1,2],2,I))
        #wantedZZ=wantedZZ/wantedZZ[0,0]
        #m=wantedZZ
        #print(x[0])
#        print(m.shape)
#        print(m1.shape)
        def makeopmat(pairs):
                opmats=[eye(3)]*numAtoms
                for pair in pairs:
                    opmats[pair[0]]=m
                    opmats[pair[1]]=None
                opmats=list(filter(lambda x: x is not None, opmats))
                opmat=spkronl(opmats)
                return opmat
        if odd==0 or odd==3:
            pairs=[]
            for i in range(0,int((numAtoms-2)/4)+1):
                pairs.append([4*i,4*i+1])
            evenMat1=makeopmat(pairs)
            pairs=[]
            for i in range(0,int((numAtoms-4)/4)+1):
                pairs.append([4*i+2,4*i+3])
            evenMat2=makeopmat(pairs)
        if odd==1 or odd==3:
            pairs=[]
            for i in range(0,int(((numAtoms-3)/4))+1):
                pairs.append([4*i+1,4*i+2])
            oddMat1=makeopmat(pairs)
            pairs=[]
            for i in range(0,int(((numAtoms-5)/4))+1):
                pairs.append([4*i+3,4*i+4])
            oddMat2=makeopmat(pairs)
        if odd ==0:
            return evenMat2*evenMat1
        elif odd==1:
            return oddMat2*oddMat1
        else:
            return oddMat2*oddMat1*evenMat2*evenMat1
            
def gensZ(x,odd=0,numAtoms=1):
    #DEPRECATED
    #gensZ generates the Hz time evolution instead of from the effective times, directly with an x.
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[x[1],0]
    hamilConds.Is=[lambda y,t: 0]*2
    hamilConds.NStates=3
    if odd!=3:
        hamilConds.specialCoupling=(list(zip([[0,1]],[lambda y,t: x[0]*(y[1] % 2==odd)])))
    else:
        hamilConds.specialCoupling=(list(zip([[0,1]],[lambda y,t: abs(x[0])])))
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*1*hamil1)
    return apphamil
def gensZt(tino,odd=0,numAtoms=1):
    #Generates the operator for the Hz step, 
    #tino is a list of times,
    #odd is whether the odd atoms or the even atoms get the application
    #numAtoms is number of atoms
    tino=(tino +math.pi)%(2*math.pi)-math.pi
    if tino==0:
        tino=1E-9
    tin=math.pi-abs(tino)
    rabi=zrabi
    detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
    t=2*pi/sqrt(rabi**2+detuning**2)#*np.asscalar(rand(1)/100+1)
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: 0*Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[0,detuning]
    hamilConds.Is=[lambda y,t: 0]*2
    hamilConds.NStates=3
    if odd!=3:
        hamilConds.specialCoupling=(list(zip([[1,2]],[lambda y,t: abs(rabi)*(y[1] % 2==odd)])))
    else:
        hamilConds.specialCoupling=(list(zip([[1,2]],[lambda y,t: abs(rabi)])))
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*t*hamil1)
    return apphamil

def genTest(x,even=0,numAtoms=1):
    ##DEPRECATED
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[x[1],0]
    hamilConds.Is=[lambda y,t: 0*(y[0]==0)*(y[1]==0)]*2
    hamilConds.NStates=2
    hamilConds.specialCoupling=(list(zip([[0,1]],[lambda y,t: x[0]*(y[1] % 2==even)])))
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*0.01*hamil1)
    return apphamil

    #appeff=genEffectHamil(apphamil,numAtoms)
    #return appeff
    
def genSwap(even=0,numAtoms=2):
    #Generates the hamiltonian for the swapping step of the process.
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: 100]
    hamilConds.Detuning=[0,0]
    #hamilConds.Is=3
    hamilConds.Is=[lambda y,t: 0*(y[0]==0)*(y[1]==0)]*2
    hamilConds.NStates=3
    if even:
        hamilConds.specialCoupling=(list(zip([[0,1]],[lambda y,t: pi*(y[1] % 2==0)])))
    else:
        hamilConds.specialCoupling=(list(zip([[0,1]],[lambda y,t: pi*(y[1] % 2==1)])))
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*1*hamil1)
    #appeff=genEffectHamil(apphamil,numAtoms)
    return apphamil
def spkronl(mats):
    matkron=mats[0]
    for mat in mats[1:]:
        matkron=sparse.kron(matkron,mat)
    return matkron
def print2out(text):
  #This is just a utility function that need not be used. It merely converts
  #[2 3  3 ] -> [2,3,3], therefore taking the printed version of an array to 
  #one that can be directly copied into python code
    text=text.strip()
    text=text.replace(' ',',')
    for i in range(5):
      text=text.replace(',,',',')
    text=text.replace('[,','[')
    text=text.replace(',[','[')
    text=text.replace('],',']')
    text=text.replace(',]',']')
    print(text)
def printr(mat,sig=3):
    if sparse.issparse(mat):
        mat=mat.todense()
    printmat(np.round(mat,sig))

def printEr(mat,numAtoms=4):
    if sparse.issparse(mat):
        mat=mat.todense()
    printmat(genEffH(np.round(mat,4),numAtoms))
def saveMin(x,f,accept):
    global minima
    minima.append([f,x])
    printer=logging.info
    if not(disfver is None) and not(disfver==''):
      printer(disfver+':'+str(x)+'->'+str(f))
    else:
      printer(str(x)+'->'+str(f))
    
def fullgenZZ(x):
    return genZZ(x[0:2],0)*genZZ(x[2:],1)*genZZ(x[4:6],0)
#def innerMat(B,A):
#    if not(sparse.issparse(A)):
#        A=sparse.lil_matrix(A)    
#    magn=((A.T.conjugate()*A)).diagonal().sum()
#    #return sum(abs(A-B))
#    out=float(abs((A.T.conjugate()*B/magn).diagonal().sum()-1))
#    return out
def innerMat(B,A):
    if sparse.issparse(A):
        A=A.todense()
    if sparse.issparse(B):
        B=B.todense()
    magn=np.trace((A.T.conjugate()*A))
    #return sum(abs(A-B))
    return float(abs(np.trace(A.T.conjugate()*B/magn)-1))
def saveVar(var,name):
    with open(name+'.pysave', 'wb') as f:
        pickle.dump(var, f)
def loadVar(loc):
    with open(loc, 'rb') as f:
       var = pickle.load(f)  
    return var
def sortOp(a):
    if isinstance(a[0],list):
      al=list(map(lambda a: a[0],a))
    else:
      al=list(map(lambda a: a.fun,a))
    return sortls(al,a)
def sortls(l1,l2):
    indexes = list(range(len(l1)))
    indexes.sort(key=l1.__getitem__)
    l1=list(map(l1.__getitem__, indexes))
    l2=list(map(l2.__getitem__, indexes))
    return l1,l2
    #Z = [x for _,x in sorted(zip(l2,l1))]
def genOptiHad():
    ##DEPRECATED
    #Was used in testing optimization for hadamard method
    wanted=1/sqrt(2)*sparse.csc_matrix([[1,1],[1,-1]])
    wanted=wanted
    Hfull=lambda x: genTest(x[0:2])*genTest(x[2:4])
    cost=lambda appeff:innerMat(appeff,wanted)
    f=lambda x:cost(Hfull(x))
    disf=lambda x,f=f:dispFunc(f,x)
    ind=(lhs(4,50)*2*math.pi).tolist()
    if __name__ == '__main__':
        with Pool(28) as p:
            a=list(map(lambda x: optimize.minimize(disf,x,bounds=[(0,20),(-20,20)]*2,method='TNC'),ind))
    al=list(map(lambda x: x.fun,a))
    asorted=sort2list(al,a)
    optia=asorted[1][0]
    return optia
#def optiHz(t):
#    wantedZ=expm(-1j*t*Z)
#    wantedZ=wantedZ/wantedZ[0,0]
#    Hfull=lambda x: gensZ(x[0:2])
#    cost=lambda appeff:innerMat(appeff,wantedZ)
#    f=lambda x:cost(Hfull(x))
#    disf=lambda x,f=f:dispFunc(f,x)
    #return optia
def optiHzz(t,numAtoms=2):
    ##DEPRECATED
    wantedZZ=expm(-1j*t*setToHilbU([Z,Z],[1,2],numAtoms,I))
    wantedZZ=wantedZZ/wantedZZ[0,0]
    Hfull=lambda x: genZZ(x[0:2],0)*genZZ(x[0:2],1)
    cost=lambda appeff:innerMat(appeff,wantedZZ)
    f=lambda x:cost(genEffH(Hfull(x)))
    disf=lambda x,f=f:dispFunc(f,x)
    ind=(lhs(2,100)*[3*pi,1]).tolist()
    if __name__ == '__main__':
        with Pool(28) as p:
            #bnds=[(0,50,),(-30,30,)]*3
            a=list(p.map(lambda x: optimize.minimize(disf,x,method='Powell'),ind))
    print(a)
    al=list(map(lambda x: x.fun,a))
    asorted=sort2list(al,a)
    return asorted[1][0]

def genInterp(steps,load=1,save=1):
    #Generate the interprepolation file that is used by the indirect approximate optimization.
    #This function will check if the file exists and if it does it will load from it, 
    #and other wise it will generate a new one. 
    #Steps=number of steps to use in the interpolation.
    #save is whether to save
    #load is whether to load
    if load and os.path.exists('Interps/hzzInterp.pysave'):
        [X,ys]=loadVar('Interps/hzzInterp.pysave')
        return lambda x: np.interp(x, X, ys)
    ys=[]
    X=np.linspace(-pi*0.9999,pi*0.9999,steps)
    global zzrabi
    for i in X:
        numAtoms=2
        tino=i
        tino=(tino +math.pi)%(2*math.pi)-math.pi
        
        tin=math.pi-abs(tino)
        rabi=zzrabi
        detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
        x=[rabi,detuning]
        #printEr(genZZ(x,0))
        Hz=genEffH(genZZ(x,0),2)
#        top=Hz[1,1]*Hz[2,2]
#        bot=Hz[1,1]
#        act=Hz[3,3]
        ys.append(cmath.phase(Hz[3,3]))
    pimod=0
    for i in range(1,len(ys)):
        ys[i]=ys[i]+2*pi*pimod
        if (ys[i]-ys[i-1])>pi/2:
            pimod=pimod-1
            ys[i]=ys[i]-2*pi
        elif (ys[i]-ys[i-1])<-pi/2:
            pimod=pimod+1
            ys[i]=ys[i]+2*pi
    interp=lambda x: np.interp(x, X, ys)
    if save:
        saveVar([X,ys],'Interps/hzzInterp')
    return interp
def genZZs(tino):
    tino=(tino +math.pi)%(2*math.pi)-math.pi
    tin=math.pi-abs(tino)
    rabi=10
    detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
    m=genZZ([rabi,detuning],0)
    return genEffH(m).todense()
def speedHzz(tino,even=0,numAtoms=2):
    ##DEPRECATED
    #Faster version of Hzz computation was used in testing.
    global interp
    Hzz=0j*eye(4,4)
    Hzz[0,0]=1
    tino=(tino +math.pi)%(2*math.pi)-math.pi
    tin=math.pi-abs(tino)
    if even==0:
        Hzz[1,1]=-exp(-1j*2*tino)
        Hzz[2,2]=exp(-1j*1*tino)
    else:
        Hzz[1,1]=-exp(-1j*1*tino)
        Hzz[2,2]=exp(-1j*2*tino)
    if interp is None:
        interp=genInterp(100)
    Hzz[3,3]=exp(1j*interp(tino))
    return Hzz
def speedZZF(tin,odd=0,numAtoms=2):
    #genZZF takes in a Detuning and Rabi Frequency x[0] and x[1] and whether to
    #run on odd atoms, even or for odd==3 both
    #For 4 atoms the process is hard coded in, however for 4+ atoms the a general
    #code is used. The general code does not work with 4 atoms or lower but inessence
    #does the same thing as the 4 atom case.
    m=speedHzz(tin,0)
    if numAtoms==4:
        if odd==0:
            return spkronl([m,eye(2),eye(2)])*spkronl([eye(2),eye(2),m])
        elif odd==1:
            return spkronl([eye(2),m,eye(2)])
        else:
            return spkronl([eye(2),m,eye(2)])*spkronl([m,eye(2),eye(2)])*spkronl([eye(2),eye(2),m])
    else:
        def makeopmat(pairs):
                opmats=[eye(2)]*numAtoms
                for pair in pairs:
                    opmats[pair[0]]=m
                    opmats[pair[1]]=None
                opmats=list(filter(lambda x: x is not None, opmats))
                opmat=spkronl(opmats)
                return opmat
        if odd==0 or odd==3:
            pairs=[]
            for i in range(0,int((numAtoms-2)/4)+1):
                pairs.append([4*i,4*i+1])
            evenMat1=makeopmat(pairs)
            pairs=[]
            for i in range(0,int((numAtoms-4)/4)+1):
                pairs.append([4*i+2,4*i+3])
            evenMat2=makeopmat(pairs)
        if odd==1 or odd==3:
            pairs=[]
            for i in range(0,int(((numAtoms-3)/4))+1):
                pairs.append([4*i+1,4*i+2])
            oddMat1=makeopmat(pairs)
            pairs=[]
            for i in range(0,int(((numAtoms-5)/4))+1):
                pairs.append([4*i+3,4*i+4])
            oddMat2=makeopmat(pairs)
        if odd ==0:
            return evenMat2*evenMat1
        elif odd==1:
            return oddMat2*oddMat1
        else:
            return oddMat2*oddMat1*evenMat2*evenMat1
def speedZ(tino,odd=0,numAtoms=1):
    Hz=eye(2,2)+0j
    tino=(tino +math.pi)%(2*math.pi)-math.pi
    tin=math.pi-abs(tino)
    Hz[1,1]=exp(-1j*1*tino)
    mats=[eye(2,2)]*numAtoms
    if odd==0:
        for i in range(numAtoms):
            if i%2 ==0:
                mats[i]=Hz
    elif odd==1:
        for i in range(numAtoms):
            if i%2 ==1:
                mats[i]=Hz
    return spkronl(mats)
def speedX(tin,numAtoms=1):
    
    Hx=expm(-1j*tin/2*X).todense()
    return spkronl([Hx]*numAtoms)
def directOPQAOA(times,parallel=0,ver=7):
    #The direct QAOA but for an operator (such as the error correcting circuit) instead of a state.
    def hamil(indexTime,numAtoms=numAtoms):
        if isinstance(indexTime,tuple):
            index=indexTime[0]
            time=indexTime[1]
        else:
            index=indexTime
            time=times[index]
        if ver==7:
            step=index % 5
            if step ==0:  
                tH=speedX(time,numAtoms)
                
            elif step==1:  
                tH=speedZ(time,0,numAtoms)

                
            elif step==2:
                tH=speedZ(time,1,numAtoms)

                
            elif step ==3:
                tH=speedZZF(time,0,numAtoms)

            elif step ==4: 
                
                tH=speedZZF(time,1,numAtoms)     
        return tH
    if parallel:
        if __name__ == '__main__':
            #t1=time.time()
            thamil=lambda indexTime,numAtoms=numAtoms:hamil(indexTime,numAtoms)
            tHs=list(p.map(thamil,zip(list(range(0,len(times))),times)))
            #t2=time.time()
            #print('HamilTime:',t2-t1)
    else:
        tHs=list(map(hamil,range(len(times))))

    Hs=tHs[0]
    #t1=time.time()
    for tH in tHs[1:]:
        Hs=tH*Hs
    return Hs
def indirectQAOA(times,psiI,ver=7,parallel=1):
    def hamil(indexTime,numAtoms=numAtoms):
        if isinstance(indexTime,tuple):
            index=indexTime[0]
            time=indexTime[1]
        else:
            index=indexTime
            time=times[index]
        if ver==7:
            step=index % 5
            if step ==0:  
                tH=speedX(time,numAtoms)
                
            elif step==1:  
                tH=speedZ(time,0,numAtoms)

                
            elif step==2:
                tH=speedZ(time,1,numAtoms)

                
            elif step ==3:
                tH=speedZZF(time,0,numAtoms)

            elif step ==4: 
                
                tH=speedZZF(time,1,numAtoms)
        elif ver==8:
            step=index % 3
            if step ==0:  
                tH=speedX(time,numAtoms)  
            elif step==1:  
                tH=speedZ(time,0,numAtoms)*speedZ(time,1,numAtoms)
            elif step ==2:
                tH=speedZZF(time,0,numAtoms)*speedZZF(time,1,numAtoms)  
        return tH
    if parallel:
        if __name__ == '__main__':
            thamil=lambda indexTime,numAtoms=numAtoms:hamil(indexTime,numAtoms)
            tHs=list(p.map(thamil,zip(list(range(0,len(times))),times)))
    else:
        tHs=list(map(hamil,range(len(times))))
    #t1=time.time()
    ys=psiI
    for tH in tHs:
        ys=tH*ys
    return ys

def getRealTimes(appTimes):
    #Converts the effective times to real times, along with detunings.
    #The rabi frequency is hard coded to 10, this is a debug function.
    ts=[]
    detunings=[]
    global zzrabi
    for i in range(len(appTimes)):
        time=appTimes[i]
        step= i % 5
        if step==0:
            t=(time)%(2*math.pi)
            detunings.append(0)
            print ('x:',t)
            ts.append((t))
        elif step==1 or step==2:
            tino=time
            tino=(tino +math.pi)%(2*math.pi)-math.pi
            if tino==0:
                tino=1E-9
            tin=math.pi-abs(tino)
            rabi=zzrabi
            detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
            detunings.append(detuning)
            t=2*pi/sqrt(rabi**2+detuning**2)
            print ('z:',2*t)
            ts.append(2*t)
        elif step==3 or step==4:
            tino=time
            tino=(tino +math.pi)%(2*math.pi)-math.pi
            if tino==0:
                tino=1E-9
            tin=math.pi-abs(tino)
            rabi=10
            detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
            detunings.append(detuning)
            t=2*pi/sqrt(rabi**2+detuning**2)
            print ('zz:',2*t)
            ts.append(2*(t+2*pi/rabi))
    return [ts,detunings]

            
def truncTimes(times):
    #Convert arbitrary set times such that they are between -pi and pi
    #so truncTimes([2*pi])=
    times=times.copy()
    trun=lambda x: (x +math.pi)%(2*math.pi)-math.pi
    for i in range(len(st)):
        if i %5==0:
            times[i]=times[i] % (2*pi)
            #st[i]=trun(st[i])
        else:
            times[i]=trun(st[i])
    return times
def convToRydberg(times):
    results=[]
    errors=[]
    for index in range(len(times)):
        t=times[index]
        subIndex=index % 5
        Hzs=[0,1]
        Hzzs=[2,3]
        if  subIndex in Hzs:
            results.append(t)
            errors.append(0)
            #ta=optiHz(t)
            #results.append(ta)
            #error=ta.fun
            #errors.append(ta.fun)
        elif subIndex in Hzzs:
            ta=optiHzz(t)
            results.append(ta)
            errors.append(ta.fun)
        else:
            results.append(2*t)
            errors.append(0)
    return [results,errors]

def conv(x):
    #Converts output types
    if isinstance(x,scipy.optimize.optimize.OptimizeResult):
        return x.x
    else:
        return x

def elongate(x):
    lo=[]
    for l in x:
        if not(np.isscalar(l)):
            lo=lo+list(l)
        else:
            lo.append(l)
    return lo    
    

def makeInd(bnds,iter):
    ind=lhs(len(bnds),iter)
    sub=[]
    mult=[]
    for bnd in bnds:
        sub.append(bnd[0])
        mult.append(bnd[1]-bnd[0])
    ind=ind*mult+sub
    return ind.tolist()


def getBnds(ver):
    #Bounds for different optimization input parameters, only version 7 and 8 are relevant.
    #7 is the full scheme, while 8 is the symmetric scheme, with no difference between odd and even steps.
    if ver ==0:
        bnds=[(0, 31.4),(-125.4, 125.4),(0, 31.4),(-125.4, 125.4),(0,2*math.pi)]
    elif ver ==1:
        bnds =[(-125.4, 125.4),(0, 31.4),(-125.4, 125.4),(0, 31.4),(-125.4, 125.4),(0, 31.4),(-125.4, 125.4),(0, 2*pi)]
    elif ver==2:
        bnds=[(0, 31.4),(-125.4, 125.4),(0,2*math.pi)]
    elif ver==3:
        bnds=[(-math.pi,math.pi),(-math.pi,math.pi),(0, 31.4),(-125.4, 125.4),(0, 31.4),(-125.4, 125.4),(0,2*math.pi)]
    elif ver==4:
        bnds=[(0, 31.4),(-125.4, 125.4),(0, 2*pi)]
    elif ver==5:
        bnds=[(-math.pi,math.pi),(-math.pi,math.pi),(-math.pi,math.pi),(-math.pi,math.pi),(0,2*math.pi)]
    elif ver==6:
        bnds=[(-math.pi,math.pi),(-math.pi,math.pi),(0,2*math.pi)]
    elif ver==7:
        bnds=[(0,2*pi),(-math.pi,math.pi),(-math.pi,math.pi),(-math.pi,math.pi),(-math.pi,math.pi)]
    elif ver==8:
        bnds=[(0,2*pi),(-math.pi,math.pi),(-math.pi,math.pi)]
    return bnds

global p

#Here we have different separated sections for the different things to optimize for.
#There are different runs=0 or run=1 for different parts
#------------------------------------------------------------------------------
#Quantum Error correcting circuit Optimization
with Pool(8) as p:
    numAtoms=5
    obState=StateRepo.errorgate
    #Direct is whether one is using the approximate method of the full method
    direct=1
    if direct ==0:
      cost=lambda op:innerMat(obState,op)
      f=lambda x:cost(directOPQAOA(x))
      disf=lambda x,f=f:dispFunc(f,x)
    else:
      cost=lambda op:innerMat(obState,genEffH(op))
      f=lambda x:cost(directQAOA(x,0,7,1,1))
      disf=lambda x,f=f:dispFunc(f,x)
    bnds=getBnds(7)*20
    #rand(len(bnds))
    ind=makeInd(bnds,100)
    lb=list(map(lambda x: x[0],bnds))
    ub=list(map(lambda x: x[1],bnds))
    #ind[0]=[rand(len(bnds))*(np.array(ub)-np.array(lb))+np.array(lb)]
    def fp(x):
        fpt=lambda x: cost(directOPQAOA(x))
        return dispFunc(fpt,x)
    run=0
    generate=1
    if run==1:
        if generate==1:
            disfver='Errorgate'
            #for i in range(len(ind)):
                #a00=PDE(fp, bnds,maxiters=1000).solve(show_progress=True)
                #a00=pyswarm.pso(disf,lb,ub,maxiter=1000)
    #        a00=optimize.basinhopping(disf,ind[0],callback=saveMin,niter=100)
    #       a00=optimize.shgo(disf,bnds,callback=saveMin)
            a00=optimize.dual_annealing(disf,bnds,callback=saveMin)
    #        a00=optimize.differential_evolution(disf,bnds,callback=saveMin)
            saveVar(a00,'Exports/QAOA/DAOpt-ErrorGate')
            a01=optimize.basinhopping(disf,a00.x,callback=saveMin,niter=100)
        else:
          #Continuation of Previous run
          disfver='Errorgate'
          st=[1.68425395e+00,1.30688031e+00,-1.04350698e+01,-8.57896276e-02
          ,-6.94392747e+00,1.10915838e+01,9.18873756e+00,5.44444250e+00
          ,-7.42005771e+00,-4.82756092e+00,1.55600597e+00,-3.84639384e+00
          ,-1.55091680e+01,6.29264548e+00,-9.94639609e+00,4.60954488e+00
          ,-3.35546300e+00,4.20845431e+00,-8.08302118e+00,3.11796161e+00
          ,4.36215059e+00,7.95655439e-01,9.82954894e+00,-3.32974501e+00
          ,2.93383242e+00,3.44168378e+00,-4.12523759e+00,4.31830277e+00
          ,6.59418042e+00,-1.59105416e+01,1.54361028e+00,4.00602015e+00
          ,1.87449626e+01,-8.13963597e+00,5.89541475e+00,5.00550436e+00
          ,-6.88837465e+00,-2.37119979e+01,1.29468298e+01,8.66985886e+00
          ,2.84151122e+00,1.22509557e+01,1.96931215e+00,8.94096766e+00
          ,-5.79212939e+00,4.91879504e+00,-7.13798520e+00,1.28562153e+01
          ,-6.60719052e+00,7.74275513e+00,3.53013811e+00,6.44177062e+00
          ,1.82217623e+01,-1.23333518e+01,-4.73879700e+00,-1.84391783e+00
          ,6.62609166e-01,-1.54763106e+01,1.54001324e+01,-1.01758473e-01
          ,-5.10044546e+00,9.36322916e+00,1.10915767e+01,-6.32563659e+00
          ,2.25260779e-02,-1.26291785e+00,6.78257948e+00,-3.46052223e+00
          ,3.23811128e-01,2.78162548e+00,5.26718477e+00,-1.26171311e+01
          ,-3.32854458e+01,2.28308523e+01,-3.93297196e+00,3.13922768e+00
          ,3.95184784e+00,2.43072339e+01,-1.04523383e+01,3.92896596e+00
          ,1.23491440e+00,-6.04441168e+00,6.19350328e+00,6.25358212e+00
          ,5.37467782e+00,-7.86248876e-01,2.95970551e+00,-6.23357502e+00
          ,-3.16503561e+00,3.36435941e+00,5.50948732e+00,1.27878693e+01
          ,-2.53780201e+00,-3.84347182e-01,-6.27404843e+00,3.91817145e+00
          ,4.39313599e+00,-5.28560878e+00,2.26136911e+00,-1.56442637e-02]
          
          a00=optimize.basinhopping(disf,st,callback=saveMin,niter=100)
          #a000=optimize.minimize(disf,st,method='POWELL')





#    with open('Exports/subOptimize/8Apps'+str(numAtoms)+'-'+str(i)+'-AME5-RydbergQAOA-ImprovedHz-0Start-'+str(int(time.time()))+'.pysave', 'wb') as f:
#        pickle.dump(opout, f)




#GHZ State Optimization  
with Pool(4) as p:
    numAtoms=5
    #psiI=setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])).todense()
    gState=setToHilbU(np.array([[1,0,0]]).T,[1],numAtoms,np.array([[1,0,0]]).T)
    #Direct is whether one is using the approximate method or the full method. 
    direct=0
    pulsever=8
    if not(direct):
        psiI=genEffState(setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])).todense())
        obState=genEffState(((setToHilbU(np.array([[0,1,0]]),[1],numAtoms,np.array([[0,1,0]])).T+gState)*1/math.sqrt(2))).getH()
        cost=lambda ys,obState=obState: -sum(abs(obState*ys)**2)
        f=lambda x,parallel=0,psiI=psiI,cost=cost: cost((indirectQAOA(parseInput(x,pulsever),psiI,pulsever,parallel)))
        dynf=lambda x,parallel=0,psiI=psiI,cost=cost: (indirectQAOA(parseInput(x,pulsever),psiI,pulsever,parallel))
    else:
        obState=((setToHilbU(np.array([[0,1,0]]).T,[1],numAtoms,np.array([[0,1,0]]).T)+gState)*1/math.sqrt(2)).getH()
        psiI=setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]]))
        cost=lambda ys,obState=obState: -sum(abs(obState*ys)**2)
        f=lambda x,parallel=1,psiI=psiI,cost=cost: cost((directQAOA(parseInput(x,pulsever),psiI,pulsever,parallel)))
        dynf=lambda x,parallel=1,psiI=psiI,cost=cost: (directQAOA(parseInput(x,pulsever),psiI,pulsever,parallel))
    disf=lambda x,f=f: dispFunc(f,x)
    def fp(x):
        fs=lambda x: f(x,0)
        return dispFunc(fs,x)
    bnds=getBnds(pulsever)*6
    lb=list(map(lambda x: x[0],bnds))
    ub=list(map(lambda x: x[1],bnds))
    ind=makeInd(bnds,100)
    typeofrun='annealB'
    run=1
    if run==1:
        disfver='GHZ-'+str(direct)
        if typeofrun=='annealB':
            a10=optimize.dual_annealing(disf,bnds,callback=saveMin)
    #        a00=optimize.differential_evolution(disf,bnds,callback=saveMin)
            saveVar(a10,'Exports/QAOA/DAOpt-ver7-GHZ')
            a11=optimize.basinhopping(disf,a10.x,callback=saveMin,niter=100)
        elif typeofrun=='bb':
            p.terminate()
            p=None
            
            a=bb.search(f=fp,  # given function
              box=bnds,  # range of values for each parameter (2D case)
              n=500,  # number of function calls on initial stage (global search)
              m=20,  # number of function calls on subsequent stage (local search)
              batch=5,  # number of calls that will be evaluated in parallel
              resfile='output.csv')  # text file where results will be saved
        elif typeofrun=='continue':
            st=[1.56942778,-2.73818117,2.99664617,-2.35470818,15.98498056,-3.20489481
              ,3.14315246,-2.86270118,-3.20710609,3.92745947,0.52994471,-3.34000893
              ,5.49894979,3.25602242,3.07350632,9.42594902,3.40634481,-3.2845338
              ,3.92706807,-2.96511505,3.09080959,4.71402295,2.63532735,6.03382085]
            st=[1.57080836,-21.58938708,14.04562439,3.14161941,12.968122
            ,-4.80393761,1.57078397,-7.84831993,1.47243185,-3.14161004
            ,1.31675997,-6.29155909,1.57079216,1.47393291,9.31992624
            ,9.42477836,-1.66770349,-9.52959919,1.57079712,-28.00772878
            ,4.62236974,-1.57079843,0.920481,0.83748761]
            #a10=optimize.minimize(disf,st,method='Nelder-Mead')
            a10=optimize.basinhopping(disf,st,callback=saveMin)
#    a10=[]
#    for i in range(len(ind)):
#        a=optimize.minimize(disf,ind[i],bounds=bnds)
#        a10.append(a)
st=[ 2.94559903e+00, -1.68077766e+00,  5.49216002e-01, -1.71712503e+00,
        1.71794266e+00,  1.75554957e-01,  2.52156145e+00, -2.21745502e-01,
        1.69651623e+00,  2.83233181e+00, -6.06506758e-01,  3.02776449e+00,
        1.50363665e+00, -1.74734285e+00,  5.60907654e+00, -3.80880962e-01,
       -3.04505292e+00, -2.64897102e+00, -5.24949541e-01,  5.30711086e+00,
        3.11683388e+00, -3.14159265e+00,  1.57388707e+00, -2.58018920e+00,
        2.29277388e+00, -3.03652040e-03, -1.99865944e-01,  8.23841874e-01,
       -1.42615449e+00,  5.46864176e+00]
#--------------------
#Cluster State Optimization
with Pool(8) as p:
    numAtoms=5
    obState=StateRepo.clusterState5.conjugate().transpose()
    direct=1
    pulsever=8
    if not(direct):
        psiI=genEffState(setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])))
        cost=lambda ys,obState=obState: -sum(abs(obState*ys)**2)
        f=lambda x,psiI=psiI,cost=cost: cost((indirectQAOA(parseInput(x,pulsever),psiI,pulsever)))
    else:
        psiI=setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])).todense()
        cost=lambda ys,obState=obState: -sum(abs(obState*genEffState(ys))**2)
        f=lambda x,psiI=psiI,cost=cost: cost((directQAOA(parseInput(x,pulsever),psiI,pulsever)))
    disf=lambda x,f=f: dispFunc(f,x)
    def fp(x):
        f=lambda x,psiI=psiI,cost=cost: cost((directQAOA(parseInput(x,7),psiI,7,0)))
        return dispFunc(f,x)
    bnds=getBnds(pulsever)*8
    lb=list(map(lambda x: x[0],bnds))
    ub=list(map(lambda x: x[1],bnds))
    ind=makeInd(bnds,100)
    #a2=PDE(fp, getBnds(6)*5,maxiters=10000).solve(show_progress=True)
    run=0
    if run==1:
      disfver='Cluster'+str(direct)
      #Generate run or continue from previous run.
      generate=0
      if generate:
        a20=optimize.dual_annealing(disf,bnds,callback=saveMin)
#        a00=optimize.differential_evolution(disf,bnds,callback=saveMin)
        #saveVar(a20,'Exports/QAOA/DAOpt-ver-8-Cluster')
        #a21=optimize.basinhopping(disf,a20.x,callback=saveMin,niter=100)
      else:
          #Continuation of previous run
          #Set the start to continue from:
        st=[  1.57079725,   8.10809241,  -3.24642536,   2.35484743,
         -5.09296471,  -3.33146268,  -3.14159088,  -8.40499916,
          9.40497385,   0.78533909,   4.73362557,   6.26889148,
          6.22302652,   6.4578968 ,  -4.80349422,   4.71236854,
         13.20092972,  -3.24642874,   3.1415932 ,  10.05933084,
         -3.24642674,   1.57079013, -13.86743699,   4.6223735 ]

        a21=optimize.basinhopping(disf,st,callback=saveMin,niter=100)
        #a21=optimize.minimize(disf,st,method='Powell')
        


        
#--------------------
#99.5% GHZ state 99.0% for new case
[ 3.56235283,  0.52638421,  1.03588289,  0.89170707,  0.52075583,
        1.21555187,  0.99623218,  0.99820104,  0.98177067,  1.23718994,
        1.08559806,  0.69485837,  0.90024389,  1.0674461 ,  0.63372224,
        0.30461217,  1.12504168,  0.86341684,  0.96141729, -0.01220621,
        0.58615866,  1.76356411,  0.98919853,  0.99422365,  0.805108  ,
        1.16803204,  0.4322642 ,  0.97472249,  1.00492848,  1.12228439]

#ClusterState with 99.87% 99.5%(for new case)
[ 2.7660041 , -6.32525028,  1.92802143,  0.99159604,  2.04470647,
        1.10029798,  1.41386036,  1.15627155,  0.86360286,  1.62281793,
        1.03336482,  0.50141098,  1.08421022,  1.01254298,  0.70931236,
        0.46381175,  1.20354213,  0.8965294 ,  1.06248818,  0.70054747,
        1.15260184,  0.39772089,  1.01821848,  1.01202006,  1.1401882 ,
        1.07870972,  0.47403393,  0.99102992,  0.98205505,  0.6694709 ]

#Process for Optimization of AME6 state.
with Pool(20) as p:
    numAtoms=6
    obState=StateRepo.AME6.conjugate().transpose()
    direct=0
    pulsever=7
    if direct:
        psiI=setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])).todense()
        gState=setToHilbU(np.array([[1,0,0]]).T,[1],numAtoms,np.array([[1,0,0]]).T)
        cost=lambda ys,obState=obState: -sum(abs(obState*genEffState(ys))**2)
        dynf=lambda x: directQAOA(parseInput(x,pulsever),psiI,pulsever)
        f=lambda x,cost=cost: cost((directQAOA(parseInput(x,pulsever),psiI,pulsever)))
    else:
        psiI=genEffState(setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])))
        cost=lambda ys,obState=obState: -sum(abs((obState*ys))**2)
        dynf=lambda x: indirectQAOA(parseInput(x,pulsever),psiI,pulsever)
        f=lambda x,parallel=1,cost=cost: cost((indirectQAOA(parseInput(x,pulsever),psiI,pulsever,parallel)))
    disf=lambda x,f=f: dispFunc(f,x)
    def fp(x):
        fs=lambda x: f(x,0)
        return dispFunc(fs,x)
    bnds=getBnds(pulsever)*8
    ind=makeInd(bnds,100)

    run=0
    if run==1:
        disfver='AME6-20-'+str(direct)
        generate=1
        if generate:
            a30=optimize.dual_annealing(disf,bnds,callback=saveMin)
            saveVar(a30,'Exports/QAOA/DAOpt-AME6-20')
            a31=optimize.basinhopping(disf,a30.x,callback=saveMin,niter=100)
        else:
            #Continuation of previous run
            #Set the start to continue from: 
            st=[  0.9302044 ,  -1.46030539,  -4.55535369,  -0.0139353 ,
            -3.03252217,   4.42977466,  -0.36995238,   4.78917356,
            -0.01393377,   2.1731934 ,   1.40073818,   3.67336716,
            -3.06059929,   3.03676788,   0.88609284,   1.57079693,
            -2.27157829,   3.16619204,  -3.24643225,  -4.95722594,
             2.22366315,  -4.5672336 , -10.48763914,   6.26925068,
            -1.83054783,   0.74234101,   3.64622623,  -0.20777862,
            -0.01393653,  -1.65340591,   2.23506717,  -1.8989153 ,
            -0.73086377,  -0.01393712,   4.52306157,   1.16115613,
             0.26000566,  -0.60499534,   3.03675665,   0.35754938]
            #a31=optimize.basinhopping(disf,st,callback=saveMin,niter=100)
            a31=optimize.minimize(disf,st,method='Powell')
            
            
            
#------------------------------------------------------------------------------
 
#st=[1.57018765,8.11008932,-3.25074604,2.36294202,-5.11385023
#,-3.32895604,-3.1433517,-8.40017294,9.4077575,0.78232634
#,4.7509745,6.28025098,6.22693141,6.43813779,-4.8259502
#,4.71334952,13.19502598,-3.25204668,3.14507373,10.0517592
#,-3.24534575,1.57537748,-13.84385581,4.60856322]
#st=[ 3.33615291e+00,  3.03259738e+00, -7.58656839e+00, -8.90247571e-02,
#    3.03566411e+00,  1.54715744e+00,  1.30810219e+01,  1.14447336e+01,
#   -3.14338196e+00, -8.81280608e+00,  5.07498896e+00, -8.24290154e+00,
#    3.68709649e-01, -2.60490222e-02, -5.94664983e-01,  1.92049058e+00,
#   -2.62856671e+00,  1.31411343e+00,  6.29956936e+00, -2.53418424e+00,
#   -1.69995928e+00, -3.54853138e+00,  2.83652924e+00, -3.90565861e+00,
#    3.72962122e-01,  2.24885739e+00, -2.82786486e+00, -2.22997994e+00,
#    2.93252067e+00,  6.69146426e-02,  2.54079844e+00, -2.02537876e+00,
#   -1.33429520e+00,  1.06598158e+00, -1.72174257e+00,  2.44550050e-02,
#    5.60236620e+00,  7.73379617e+00, -3.96781910e+00,  1.46956965e+00,
#    3.97220537e+00, -4.47928671e+00,  4.32386031e+00, -3.15452252e+00,
#    1.08699661e-02,  4.79291787e+00, -2.34836189e+00,  2.39457135e+00,
#   -3.31548248e+00,  2.99443796e+00,  4.68086115e+00,  2.40927310e-01,
#    1.31500739e-01, -4.89817477e-03, -3.27400057e+00,  4.71961185e+00,
#    8.35483315e+00,  4.62851176e+00, -4.78067159e+00, -3.23211991e+00,
#    2.22238827e+00,  1.70074569e+00, -7.74600528e+00,  6.29944689e+00,
#   -1.05014951e-01,  1.52410467e+00,  4.01194224e+00, -2.12833113e+00,
#   -4.33128676e-03, -3.22660642e+00,  4.74964805e+00, -1.48053857e+01,
#   -3.53939245e-01, -3.35447645e+00,  9.01380276e+00,  6.69261422e-01,
#   -2.58937179e+00, -1.35194587e+01,  6.39582606e+00,  6.54189259e+00,
#    5.24390325e+00, -4.67783215e+00, -3.09059840e+00, -3.81685686e+00,
#   -3.33477781e+00,  4.64228992e+00, -7.56037558e-02, -6.08453380e+00,
#    3.09385215e+00,  1.53795689e-03,  6.10775791e+00,  1.11636321e+01,
#   -1.54951769e+00, -4.80176613e-02, -6.25564739e+00,  7.92243106e+00,
#    6.39972888e+00, -1.31184018e+01,  8.92783549e+00, -3.14159266e+00,
#    8.53764518e+00, -9.10107072e+00, -4.32939123e+00, -1.09114666e-01,
#    9.11746248e+00,  8.41558521e+00, -3.10576302e+00,  7.67791620e+00,
#    3.14153566e+00,  1.35506879e+00,  3.17757225e+00,  6.23642426e+00,
#    1.76980128e+01, -3.31864038e+00, -5.78182162e+00,  4.34936739e+00,
#   -4.38722429e+00, -2.17662261e+00, -6.41252308e-02,  4.54510803e+00,
#    4.57656314e+00, -2.74591161e-02, -2.93203439e+00,  3.09135587e+00,
#   -4.91626857e+00]

newst=[]
for i in range(len(st)):
  if i%5==0:
    newst.append(st[i]%(2*math.pi))
  else:
    newst.append((st[i] +math.pi)%(2*math.pi)-math.pi)
np.set_printoptions(suppress=True)
print(np.round(newst,4))   

