import numpy as np
#import matplotlib.pyplot as plt
import datetime as dt
#import corner

truth1=100
truth2=.4
truth3=.5
atruth= 1.
inctruth=60.
omgtruth=250.
oMGtruth=120.
truth=[truth1,truth2,truth3,atruth,inctruth,omgtruth,oMGtruth]

banyaklangkah=20
Pda=[154,50,380,200,975,678]
################################
for namafile in range(1,7):


############RANGE PARAMETER AWAL##########
      zp0=[.1*Pda[namafile-1],2*Pda[namafile-1]]
      zp1=[0.,1.]
      zp2=[0.,.99]
      zp=[zp0,zp1,zp2] #P tau e
      zmean=[]
      zsdev=[]
      for i in range (3):
          zmean.append((zp[i][1]+zp[i][0])/2)
          zsdev.append((zp[i][1]-zp[i][0])/2)
      zparam_zero=np.array([zmean,zsdev])

      zfaktor=[int(5e4),.5,.6,5] #banyaknya matriks N, nilai alpha,alpha', nilai q
      zelite=(int(0.01*zfaktor[0]))


      today=dt.datetime.today()
      date=(today.year,today.month,today.day,today.hour,today.minute,today.second)
      ####data yang akan dibaca#############################################################

      '''
      namafile/=10
      (z)=(np.genfromtxt('D:\TAda\Code\Data\Sintetis\Selected\Data-TXY-Selected-sigma-0.05-%s.txt' %namafile,unpack=True))
      #(z)=(np.genfromtxt('/home/prabowo/TA/Data/Selected/Data-TXY-Selected-sigma-0.05-%s.txt' %namafile,unpack=True))
      #(z)=(np.genfromtxt('/home/yuda/Bowo/Data/Selected/Data-TXY-Selected-sigma-0.05-%s.txt' %namafile,unpack=True))

      zdataobs=z[1],(z[1]-[3]),z[2],(z[2]-z[4]) #xobs,xerr,yobs,yerr
      zdatasint=len(z[0]),(np.std(z[1]-z[3])+np.std(z[2]-z[4]))*100/2,z[0] #N,sigma,year
      '''

      #(z)=(np.genfromtxt('D:\TAda\Code\Data\Da\d\\Data TXY Selected %s.txt' %namafile,unpack=True))
      (z)=(np.genfromtxt('/home/yuda/Bowo/Data/Data TXY Selected %s.txt' %namafile,unpack=True))
      #(z)=(np.genfromtxt('/home/prabowo/TA/Data//ad/%s.txt' %namafile,unpack=True))
      zdataobs=z[1],z[3],z[2],z[4] #xobs,xerr,yobs,yerr
      zdatasint=len(z[0]),np.sqrt((np.std(z[3],ddof=1)**2+np.std(z[4],ddof=1)**2)/2),z[0] #N,sigma,year

      ######################################################################################



      #####SEMUA DIMULAI SESUDAH INI#######################################################

      ###BIKIN SEMUA TEBAKAN


      def guestRange(param_meansdev,faktoriterasi,iterasike,param_meansdev_old): #faktoriterasi = N, alpha, q

            gp=[[],[],[]]
            if iterasike<1:
                  for i in range (faktoriterasi[0]):
                        for j in range(3):
                              random=np.random.uniform(-1,1)
                              gptoappend=param_meansdev[0][j]+param_meansdev[1][j]*random
                              gp[j].append(gptoappend)

            else:
                  ga=faktoriterasi[1]
                  gad=faktoriterasi[2]-faktoriterasi[2]*(1-1/(iterasike+1))**faktoriterasi[3]
                  gmeanpar=[]
                  gsdevpar=[]
                  for i in range (3):
                        gmeanpar.append(ga*param_meansdev[0][i]+(1-ga)*param_meansdev_old[0][i])
                        gsdevpar.append(gad*param_meansdev[1][i]+(1-gad)*param_meansdev_old[1][i])
                  for i in range (faktoriterasi[0]):
                        for j in range(3):
                              random=np.random.normal(0,1)
                              gptoappend=gmeanpar[j]+gsdevpar[j]*random
                              gp[j].append(gptoappend)
            zmean=[]
            zsdev=[]
            for j in range (3):
                  zmean.append(np.mean(gp[j]))
                  zsdev.append(np.std(gp[j]))
            op=np.array([zmean,zsdev])
            return(gp,op)


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
              B=(b*r11-c*r12)/d
              G=(-c*r11+a*r12)/d
              return(B,G)
          elif which==1:
              r21=np.sum(w*data[3]*data[0])
              r22=np.sum(w*data[3]*data[1])
              A=(b*r21-c*r22)/d
              F=(-c*r21+a*r22)/d
              return(A,F)

      '''
      lnP,tau,e=theta
      xobs,xerr,yobs,yerr=dataobs
      N,sigma,yearobs=datasint
      '''
      def thieletocampbell(A,B,F,G):
            omgplus=np.arctan((B-F)/(A+G))
            #omgplus=np.rad2deg(omgplus) #-90 - 90

            omgmin=np.arctan((-B-F)/(A-G))
            #omgmin=np.rad2deg(omgmin) #-90 - 90

            bmin=(B-F)
            bplus=-(B+F)
            if bmin>0:
                  if omgplus<0:
                        omgplus+=np.pi#*3
                  #elif omgplus>0:
                  #     omgplus+=180*2
            elif bmin<0:
                  if omgplus>0:
                        omgplus+=np.pi
                  elif omgplus<0:
                        omgplus+=np.pi*2
            if bplus>0:
                  if omgmin<0:
                        omgmin+=np.pi#*3
                  #elif omgmin>0:
                  #     omgmin+=180*2
            elif bmin<0:
                  if omgmin>0:
                        omgmin+=np.pi
                  elif omgmin<0:
                        omgmin+=np.pi*2
            omgnow=(omgplus+omgmin)/2
            oMGnow=(omgplus-omgmin)/2

            if oMGnow>np.pi:
                  oMGnow-=np.pi
                  omgnow-=np.pi
            elif oMGnow<0:
                  oMGnow+=np.pi
                  omgnow+=np.pi

            omgplus=omgnow+oMGnow
            omgmin=omgnow-oMGnow

            q1=(A+G)/np.cos(omgplus)
            q2=(A-G)/np.cos(omgmin)

            incnow=np.rad2deg(2*np.arctan(np.sqrt(q2/q1)))
            anow=(q1+q2)/2

            return([anow,incnow,np.rad2deg(omgnow),np.rad2deg(oMGnow)])

      def lnlike(theta, dataobs, datasint,i): #theta p tau e ujicoba

            Xtmp=[]
            Ytmp=[]
            for j in range (datasint[0]):
                  Et=0.1
                  dlt=1
                  err=1e-5
                  mu=2*np.pi/theta[0][i]
                  tminT=datasint[2][j]-theta[1][i]*theta[0][i]   ###ini dalam apa?
                  M=mu*tminT
                  while (abs(dlt)>err):
                      Ei =M+theta[2][i]*np.sin(Et)
                      #Eatas=(Et-e*np.sin(Et)-M)
                      #Ebawah=(1-e*np.cos(Et))
                      #Ei=Et-Eatas/Ebawah
                      dlt =Ei-Et
                      Et=Ei
                  #print(e,M,Et,i)
                  X =np.cos(Ei)-theta[2][i]
                  Y =np.sqrt(1-theta[2][i]**2)*np.sin(Ei)
                  #print(X,Y)
                  Xtmp.append(X)
                  Ytmp.append(Y)

            #print(e,tminT,tau,M)
            #print(1)
            Xtmp=np.array(Xtmp)
            Ytmp=np.array(Ytmp)

            data=Xtmp,Ytmp,dataobs[0],dataobs[2]

            Bhat,Ghat=leastSquareEstimates(dataobs[1],data,0)
            Ahat,Fhat=leastSquareEstimates(dataobs[3],data,1)

            xtheo=Bhat*Xtmp+Ghat*Ytmp
            ytheo=Ahat*Xtmp+Fhat*Ytmp

            xmd =xtheo-dataobs[0]
            ymd =ytheo-dataobs[2]

            kai=(np.sum((xmd/dataobs[1])**2)+np.sum((ymd/dataobs[3])**2))
            ln_like=-.5*kai

            return(ln_like,np.array([Ahat,Bhat,Fhat,Ghat]))

      def lnprior(theta,priori,i):
          P, tau, e= theta[0][i],theta[1][i],theta[2][i]
          if priori[0][0]< P <priori[0][1]  and priori[1][0] < tau < priori[1][1] and priori[2][0] < e < priori[2][1]:
              return 0.0
          return -np.inf

      def lnprob(theta, dataobs, datasint,pthoy):
           lnprobijk=[]
           opar=[]
           for i in range(len(theta[0])):
                 lp = lnprior(theta,pthoy,i)
                 if not np.isfinite(lp):
                       lnprobijk.append(-np.inf)
                       opar.append(np.array([0,0,0,0]))
                 else:
                       like=lnlike(theta, dataobs, datasint,i)
                       lnprobijk.append(lp + like[0])
                       opar.append(like[1])
           return(lnprobijk,opar)

      def sorting(theta,lnprobability,N_elite,thieta):
          parameter_sorted=[[],[],[],[],[],[],[]]
          parameter_elite= [[],[],[],[],[],[],[],[]]
          batasbaru=[]
          for i in range(len(lnprobability)):
                for j in range(3):
                      parameter_sorted[j].append([lnprobability[i],theta[j][i]])
                for j in range(3,7):
                      parameter_sorted[j].append([lnprobability[i],thieta[i][j-3]])
          for i in range(7):
                parameter_sorted[i].sort(reverse=True)
          for i in range(8):
                for j in range(0,N_elite):
                      if i ==0:
                            parameter_elite[i].append(parameter_sorted[i][j][0])
                      else:
                            parameter_elite[i].append(parameter_sorted[i-1][j][1])
          for i in range(3):
                medd=np.mean(parameter_elite[i+1])
                stdd=np.std(parameter_elite[i+1],ddof=1)
                batasbaru.append([medd,stdd])

          return(batasbaru,parameter_elite)

      Aparall=  [[],[],[],[],[],[],[],[]]
      Aparlite= [[],[],[],[],[],[],[],[]]
      Anw=[[],[]]

      paramtheta=zparam_zero
      paramold=0

      for i in range(banyaklangkah): #banyaknya langkah
            print(i)

            parameter,op=guestRange(paramtheta,zfaktor,i,paramold)
            probln,thiele=lnprob(parameter,zdataobs,zdatasint,zp)
            parametersorted=sorting(parameter,probln,zelite,thiele)

            paramold=op
            paramtheta=np.array(parametersorted[0]).T
            Anw[0].append(paramold)
            Anw[1].append(paramtheta)
            thiele=np.array(thiele).T
            Aparall[0].append(probln)
            for j in range(1,4):
                  Aparall[j].append(parameter[j-1])
            for j in range(4,8):
                  Aparall[j].append(thiele[j-4])

            for j in range(8):
                  Aparlite[j].append(parametersorted[1][j])


      Apara=[[],[],[],[],[],[],[],[]]
      for i in range(8):
          for j in range(len(Aparall[i])):
              for k in range(len(Aparall[i][j])):
                  Apara[i].append(Aparall[i][j][k])

      Apare=[[],[],[],[],[],[],[],[]]
      for i in range(8):
          for j in range(len(Aparlite[i])):
              for k in range(len(Aparlite[i][j])):
                  Apare[i].append(Aparlite[i][j][k])



      today2=dt.datetime.today()
      date2=(today2.year,today2.month,today2.day,today2.hour,today2.minute,today2.second)
      elapsed=[]
      for i in range(len(date)):
            elapsed.append(date2[i]-date[i])

      titleatribute=(namafile,banyaklangkah,zfaktor[0])

      iall=(Apara[1],Apara[2],Apara[3],Apara[4],Apara[5],Apara[6],Apara[7])
      ieli=(Apare[1],Apare[2],Apare[3],Apare[4],Apare[5],Apare[6],Apare[7])

      #Spare=np.array(Apare[1:8]).T[int(0.5*len(Apare[0])):len(Apare[0])]

      #fig = corner.corner(Spare,labels=["$P$", "$tau$", "$e$","$a$","$i$","$om$","$OM$"], truths=truth)


      np.savetxt("/home/yuda/Bowo/Result/CE-Elpd-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("/home/yuda/Bowo/Result/CE-All-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[iall])
      np.savetxt("/home/yuda/Bowo/Result/CE-Alite-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[ieli])

      '''
      np.savetxt("D:\TAda\Code\Result\CE\Da\CE-Elpd-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("D:\TAda\Code\Result\CE\Da\CE-All-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[iall])
      np.savetxt("D:\TAda\Code\Result\CE\Da\CE-Alite-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[ieli])

      np.savetxt("/home/prabowo/TA/Result/CE/CE-Elpd-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[elapsed])
      np.savetxt("/home/prabowo/TA/Result/CE/CE-All-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[iall])
      np.savetxt("/home/prabowo/TA/Result/CE/CE-Alite-F%g-Iterasi%g-N%g.txt"
                         %titleatribute,np.c_[ieli])
      '''