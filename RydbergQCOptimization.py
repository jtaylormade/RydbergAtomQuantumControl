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
        if ver==7:
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
    if ver==5 or ver==7:
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

def genX(time,numAtoms=4):
    #Generates the Hx time evolution step given an effective time.
    time=time#*np.asscalar(rand(1)/100+1)
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[0,0]
    hamilConds.Is=[lambda y,t:1,0]
    hamilConds.NStates=3
    hamilConds.posList=list(map(lambda x: [0,x],range(numAtoms)))
    hamilConds.numAtoms=numAtoms
    hamil1=sparse.csc_matrix(calcHamilGen.calcHamilGen(hamilConds,0))
    apphamil=expm(-1j*1*time*hamil1)
    return apphamil

def genZZ(x,even=0,numAtoms=2):
    #gensZZ generates the Hzz time evolution instead of from the effective times, directly with an x.
    #Where x is the Detuning and Rabi Frequency. 
    x=np.array(x)
    phase2=abs(x[0])
    Detuning1=x[1]
    t=2*pi/sqrt(x[1]**2+x[0]**2)#*np.asscalar(rand(1)/100+1)
    hamilConds=calcHamilGen.HamilProp()
    hamilConds.calcVijs=[0,lambda x,y: Vij*(np.linalg.norm(np.array(x)-np.array(y))**(-6))]
    hamilConds.Detuning=[0,0]
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
    #genZZF takes in a Rabi Frequency and Detuning x[0] and x[1] and whether to
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
    #gensZ generates the Hz time evolution instead of from the effective times, directly with an x.
    #Where x is the rabi frequency and detuning. 
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
    X=np.linspace(-pi*0.999999,pi*0.999999,steps)
    global zzrabi
    for i in X:
        numAtoms=2
        tino=i
        tino=(tino +math.pi)%(2*math.pi)-math.pi
        
        tin=math.pi-abs(tino)
        rabi=zzrabi
        detuning=np.sign(tino)*sqrt((tin**2*rabi**2)/(math.pi**2-tin**2))
        x=[rabi,detuning]
        Hz=genEffH(genZZ(x,0),2)
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
    #This returns the effective ZZ unitary operation.  
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
    #speedZ takes in the effective time tino and returns the effective Z unitary operation
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
    #Make a list of lists into a single list. 
    lo=[]
    for l in x:
        if not(np.isscalar(l)):
            lo=lo+list(l)
        else:
            lo.append(l)
    return lo    
    

def makeInd(bnds,iter):
    #Generates a well distributed but random set of starting points using Latin hypercube sampling
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
    if ver==7:
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
            a00=optimize.dual_annealing(disf,bnds,callback=saveMin)
            saveVar(a00,'Exports/QAOA/DAOpt-ErrorGate')
            a01=optimize.basinhopping(disf,a00.x,callback=saveMin,niter=100)
        else:
          #Continuation of Previous run
          disfver='Errorgate'
          #To continue with the basin hopping method Uncomment below and fill in st to your previous selected result.
          #st=[{YOUR_RESULT}]
          a00=optimize.basinhopping(disf,st,callback=saveMin,niter=100)
          #a000=optimize.minimize(disf,st,method='POWELL')





#    with open('Exports/subOptimize/8Apps'+str(numAtoms)+'-'+str(i)+'-AME5-RydbergQAOA-ImprovedHz-0Start-'+str(int(time.time()))+'.pysave', 'wb') as f:
#        pickle.dump(opout, f)




#GHZ State Optimization  
with Pool(4) as p:
    numAtoms=5
    gState=setToHilbU(np.array([[1,0,0]]).T,[1],numAtoms,np.array([[1,0,0]]).T) #Ground State
    #Direct is whether one is using the approximate method or the full method. 
    direct=0
    pulsever=8
    if not(direct):
        psiI=genEffState(setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]])).todense())#Initial State
        obState=genEffState(((setToHilbU(np.array([[0,1,0]]),[1],numAtoms,np.array([[0,1,0]])).T+gState)*1/math.sqrt(2))).getH()#Objective State
        cost=lambda ys,obState=obState: -sum(abs(obState*ys)**2)
        f=lambda x,parallel=0,psiI=psiI,cost=cost: cost((indirectQAOA(parseInput(x,pulsever),psiI,pulsever,parallel)))
        dynf=lambda x,parallel=0,psiI=psiI,cost=cost: (indirectQAOA(parseInput(x,pulsever),psiI,pulsever,parallel))
    else:
        obState=((setToHilbU(np.array([[0,1,0]]).T,[1],numAtoms,np.array([[0,1,0]]).T)+gState)*1/math.sqrt(2)).getH()#Objective State
        psiI=setToHilbU(sparse.coo_matrix([[1],[0],[0]]),[1],numAtoms,sparse.coo_matrix([[1],[0],[0]]))#Initial State
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
            #To continue with the basin hopping method Uncomment below and fill in your previous selected result.
#            st=[1.56942778,-2.73818117,2.99664617,-2.35470818,15.98498056,-3.20489481
#              ,3.14315246,-2.86270118,-3.20710609,3.92745947,0.52994471,-3.34000893
#              ,5.49894979,3.25602242,3.07350632,9.42594902,3.40634481,-3.2845338
#              ,3.92706807,-2.96511505,3.09080959,4.71402295,2.63532735,6.03382085]
            a10=optimize.basinhopping(disf,st,callback=saveMin)
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

newst=[]
for i in range(len(st)):
  if i%5==0:
    newst.append(st[i]%(2*math.pi))
  else:
    newst.append((st[i] +math.pi)%(2*math.pi)-math.pi)
np.set_printoptions(suppress=True)
print(np.round(newst,4))   

