# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:42:16 2019

@author: moham
"""

#import corner
import numpy as np
#import matplotlib.pyplot as plt
import random as rd
import datetime as dt



truth1=100
truth2=.4
truth3=.5
atruth= 1.
inctruth=60.
omgtruth=250.
oMGtruth=120.
truth=[truth1,truth2,truth3,atruth,inctruth,omgtruth,oMGtruth]
Pda=[154,50,380,200,975,678]
for namafile in range (4,5):
      today=dt.datetime.today()
      date=(today.year,today.month,today.day,today.hour,today.minute,today.second)

      ############RANGE PARAMETER AWAL##########
      zp0=[10,100]
      zp1=[0.,1.]
      zp2=[0.,.99]
      zp=[zp0,zp1,zp2] #P tau e

      ggmma=2.38/np.sqrt(2*3)

      w0=.01*(zp0[1]-zp0[0])
      w1=.01*(zp1[1]-zp1[0])
      w2=.01*(zp2[1]-zp2[0])
      wv=[w0,w1,w2]

      nstep=100000
      nchain=10
      ################################

      ####data yang akan dibaca#############################################################
      namafile/=10
      '''

      (z)=(np.genfromtxt('D:\TAda\Code\Data\Sintetis\Selected\Data-TXY-Selected-sigma-0.05-%s.txt' %namafile,unpack=True))
      #(z)=(np.genfromtxt('/home/prabowo/TA/Data/Selected/Data-TXY-Selected-sigma-0.05-%s.txt' %namafile,unpack=True))


      (z)=(np.genfromtxt('D:\TAda\Code\Data\Data WDS\letxte\%s.txt' %namafile,skip_header=1,usecols=(0,1,2,3,4),unpack=True))
      '''

      #(z)=(np.genfromtxt('/home/yuda/Bowo/Data/Data-TXY-Selected-sigma-0.05-0.%s.txt' %namafile,unpack=True))
      #(z)=(np.genfromtxt('/home/yuda/Bowo/Data/Data TXY Selected %s.txt' %namafile,unpack=True))
      (z)=(np.genfromtxt('D:\TAda\Code\Data\Sintetis\Selected\Data-TXY-Selected-sigma-0.05-%s.txt' %namafile,unpack=True))
      zdataobs=z[1],z[3]-z[1],z[2],z[4]-z[2] #xobs,xerr,yobs,yerr
      zdatasint=len(z[0]),(np.std(z[1]-z[3])+np.std(z[2]-z[4]))*100/2,z[0] #N,sigma,year
      ######################################################################################



      #####SEMUA DIMULAI SESUDAH INI#######################################################

      ###############

      ####BARU MASUK NGITUNG LIKELIHOOD

      def leastSquareEstimates(error,data,which):  #)0 AF, 1BG

          w=1/error**2
          a=np.sum(w*data[0]**2)
          b=np.sum(w*data[1]**2)
          c=np.sum(w*data[0]*data[1])
          d=a*b-c*c

          if which==0:
              r11=np.sum(w*data[2]*data[0])
              r12=np.sum(w*data[2]*data[1])
              A=(b*r11-c*r12)/d
              F=(-c*r11+a*r12)/d
              return(A,F)
          elif which==1:
              r21=np.sum(w*data[3]*data[0])
              r22=np.sum(w*data[3]*data[1])
              B=(b*r21-c*r22)/d
              G=(-c*r21+a*r22)/d
              return(B,G)

      '''
      lnP,tau,e=theta
      xobs,xerr,yobs,yerr=dataobs
      N,sigma,yearobs=datasint
      '''


      def lnlike(theta, dataobs, datasint): #theta p tau e ujicoba
            Xtmp=[]
            Ytmp=[]
            for j in range (datasint[0]):
                  Et=0.1
                  dlt=1
                  err=1e-5
                  mu=2*np.pi/theta[0]
                  tminT=(datasint[2][j]-datasint[2][0])-theta[1]*theta[0]   ###ini dalam apa?
                  M=mu*tminT
                  while (abs(dlt)>err):
                        Ei =M+theta[2]*np.sin(Et)
                        #Eatas=(Et-e*np.sin(Et)-M)
                        #Ebawah=(1-e*np.cos(Et))
                        #Ei=Et-Eatas/Ebawah
                        dlt =Ei-Et
                        Et=Ei
                  #print(e,M,Et,i)
                  X =np.cos(Ei)-theta[2]
                  Y =np.sqrt(1-theta[2]**2)*np.sin(Ei)
                  #print(X,Y)
                  Xtmp.append(X)
                  Ytmp.append(Y)

              #print(e,tminT,tau,M)
              #print(1)
            Xtmp=np.array(Xtmp)
            Ytmp=np.array(Ytmp)

            data=Xtmp,Ytmp,dataobs[0],dataobs[2]

            Ahat,Fhat=leastSquareEstimates(dataobs[1],data,0)
            Bhat,Ghat=leastSquareEstimates(dataobs[3],data,1)

            xtheo=Ahat*Xtmp+Fhat*Ytmp
            ytheo=Bhat*Xtmp+Ghat*Ytmp

            xmd =xtheo-dataobs[0]
            ymd =ytheo-dataobs[2]

            kai=(np.sum((xmd/dataobs[1])**2)+np.sum((ymd/dataobs[3])**2))

            ln_like=-.5*kai

            return(ln_like,np.array([Ahat,Bhat,Fhat,Ghat]))

      def lnprior(theta,priori):
          P, tau, e= theta
          if priori[0][0]< P <priori[0][1]  and priori[1][0] < tau < priori[1][1] and priori[2][0] < e < priori[2][1]:
              return 0.0
          return -np.inf

      def lnprob(theta, dataobs, datasint,pthoy):
           lp = lnprior(theta,pthoy)
           if not np.isfinite(lp):
                 lnprob=(-np.inf)
                 opar=(np.array([0,0,0,0]))
           else:
                 like=lnlike(theta, dataobs, datasint)
                 lnprob=(lp + like[0])
                 opar=(like[1])
           return(lnprob,opar)


      randompar=[[[],[],[]]]
      for i in range (nchain):
            for j in range (3):
                  randompar[0][j].append(rd.uniform(zp[j][0],zp[j][1]))

      Athold=[[],[],[],[],[],[],[],[]]
      Athnew=[[],[],[],[],[],[],[],[]]
      Athacc=[[],[],[],[],[],[],[]]

      intNchain=np.arange(0,nchain)
      for i in range (nstep):
            randompar.append([[],[],[]])
            for j in range(nchain):
                  intNchainnew1=np.delete(intNchain,j)
                  rp1=rd.choice(intNchainnew1)
                  intNchainnew2=np.delete(intNchainnew1,np.where(intNchainnew1==rp1)[0][0])
                  rp2=rd.choice(intNchainnew2)

                  ##newvaluetetha
                  nvthetaplus=[]
                  nvthetamin=[]

                  nvtheta=[]
                  oldtheta=[]
                  for k in range(3):
                        wvalue=np.random.normal(0,np.std(randompar[i][k]))
                        rpj12=ggmma*(randompar[i][k][rp1]-randompar[i][k][rp2])
                        newvalue=randompar[i][k][j]+rpj12+wvalue

                        #newvalue1=randompar[i][k][j]+rpj12+wvalue
                        #if newvalue1>zp[k][1]:
                        #      newvalue1=randompar[i][k][j]+.5*rpj12+wvalue
                        #newvalue2=randompar[i][k][j]-rpj12+wvalue
                        #if newvalue2>zp[k][0]:
                        #      newvalue2=randompar[i][k][j]-.5*rpj12+wvalue

                        #nvthetaplus.append(newvalue1)
                        #nvthetamin.append(newvalue2)
                        nvtheta.append(newvalue)
                        oldtheta.append(randompar[i][k][j])



                  #lnprobnew1=lnprob(nvthetaplus,zdataobs,zdatasint,zp)
                  #lnprobnew2=lnprob(nvthetamin,zdataobs,zdatasint,zp)

                  lnprobnew,nvthiel=lnprob(nvtheta,zdataobs,zdatasint,zp)
                  lnprobold,oldthiel=lnprob(oldtheta,zdataobs,zdatasint,zp)

                  #if lnprobnew1>=lnprobnew2:
                   #     nvtheta=nvthetaplus
                    #    lnprobnew=lnprobnew1
                  #else:
                   #     nvtheta=nvthetamin
                    #    lnprobnew=lnprobnew2
                  Athnew[0].append(lnprobnew)
                  Athold[0].append(lnprobold)
                  for k in range(1,4):
                        Athnew[k].append(nvtheta[k-1])
                        Athold[k].append(oldtheta[k-1])
                  for k in range(4,8):
                        Athnew[k].append(nvthiel[k-4])
                        Athold[k].append(oldthiel[k-4])

                  avalue=min(1,np.exp(lnprobnew-lnprobold))
                  urandomvalue=np.random.uniform(0,1)

                  if urandomvalue<avalue:
                        for k in range(3):
                              randompar[i+1][k].append(nvtheta[k])
                              Athacc[k].append(nvtheta[k])
                        for k in range(3,7):
                              Athacc[k].append(nvthiel[k-3])

                  else:
                        for k in range(3):
                              randompar[i+1][k].append(oldtheta[k])
                              Athacc[k].append(oldtheta[k])
                        for k in range(3,7):
                              Athacc[k].append(oldthiel[k-3])

            print(namafile,'ke',i)

      Athacc1=Athacc[0]
      Athacc2=Athacc[1]
      Athacc3=Athacc[2]
      Athacc4=Athacc[3]
      Athacc5=Athacc[4]
      Athacc6=Athacc[5]
      Athacc7=Athacc[6]
      #Athacc=np.array(Athacc).T

      Athnew0=Athnew[0]
      Athnew1=Athnew[1]
      Athnew2=Athnew[2]
      Athnew3=Athnew[3]
      Athnew4=Athnew[4]
      Athnew5=Athnew[5]
      Athnew6=Athnew[6]
      Athnew7=Athnew[7]
      #Athnew=np.array(Athnew).T

      Athold0=Athold[0]
      Athold1=Athold[1]
      Athold2=Athold[2]
      Athold3=Athold[3]
      Athold4=Athold[4]
      Athold5=Athold[5]
      Athold6=Athold[6]
      Athold7=Athold[7]
      #Athold=np.array(Athold).T

      #zuf=int(len(Athacc))#nchain*nstep)
      #zlf=int(.1*zuf)

      #samples=Athacc[zlf:zuf]
      #fig = corner.corner(samples, labels=["$P$", "$tau$", "$e$", "$a$", "$i$", "$omg$", "$oMG$"], truths=truth)

      today2=dt.datetime.today()
      date2=(today2.year,today2.month,today2.day,today2.hour,today2.minute,today2.second)
      elapsed=[]
      for i in range(len(date)):
            elapsed.append(date2[i]-date[i])


      titleatribute=(namafile,nstep,nchain)

      iold=(Athold1,Athold2,Athold3,Athold4,Athold5,Athold6,Athold7)
      inew=(Athnew1,Athnew2,Athnew3,Athnew4,Athnew5,Athnew6,Athnew7)
      icc=(Athacc1,Athacc2,Athacc3,Athacc4,Athacc5,Athacc6,Athacc7)



      '''
      np.savetxt("/home/yuda/Bowo/Result/DE-Elpd-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("/home/yuda/Bowo/Result/DE-Acc-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[icc])
      np.savetxt("/home/yuda/Bowo/Result/DE-Anew-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[inew])
      np.savetxt("/home/yuda/Bowo/Result/DE-Aold-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[iold])




      np.savetxt("D:\TAda\Code\Result\DE\WDS\DE-Elpd-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("D:\TAda\Code\Result\DE\WDS\DE-Acc-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[icc])
      np.savetxt("D:\TAda\Code\Result\DE\WDS\DE-Anew-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[inew])
      np.savetxt("D:\TAda\Code\Result\DE\WDS\DE-Aold-File-ke-%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[iold])

      np.savetxt("D:\TAda\Code\Result\DE\Sintetik\DE-Elpd-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("D:\TAda\Code\Result\DE\Sintetik\DE-Acc-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[icc])
      np.savetxt("D:\TAda\Code\Result\DE\Sintetik\DE-Anew-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[inew])
      np.savetxt("D:\TAda\Code\Result\DE\Sintetik\DE-Aold-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[iold])



      np.savetxt("/home/prabowo/TA/Result/DE/Sintetik/DE-Elpd-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("/home/prabowo/TA/Result/DE/Sintetik/DE-Acc-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[icc])
      np.savetxt("/home/prabowo/TA/Result/DE/Sintetik/DE-Anew-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[inew])
      np.savetxt("/home/prabowo/TA/Result/DE/Sintetik/DE-Aold-F%g-Nstep%g-Nchain%g.txt"
                         %titleatribute,np.c_[iold])
      '''