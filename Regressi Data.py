import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

pda=[154,50,380,200,975,678]
namafile=1
wdw=3
(ct,cx,cy,crx,cry)=(np.genfromtxt('D:\TAda\Code\Data\Da\\ad\%s.txt'
     %namafile,unpack=True))



r2x=[]
r2y=[]
xs=[]
ys=[]

for i in range(6):    #banyaknya jumlah n polinomial
    px = np.poly1d(np.polyfit(ct, cx, i)) #polinom fitting rho
    py = np.poly1d(np.polyfit(ct, cy, i)) #polinom fitting teta

    xpolinom=px(ct)
    ypolinom=py(ct)

    r2x.append(r2_score(cx,xpolinom))
    r2y.append(r2_score(cy,ypolinom))

    xs.append(xpolinom)
    ys.append(ypolinom)


r2x=np.array(r2x)
r2y=np.array(r2y)

xc=xs[np.argmax(r2x)]
yc=ys[np.argmax(r2y)]

cex=abs(cx-xc)
cey=abs(cy-yc)

at=[ct[0]]
ax=[cx[0]]
ay=[cy[0]]
aex=[cex[0]]
aey=[cey[0]]
ott=ct[0]
for i in range(len(ct)-1):
    if ct[i+1]-ott>=.4:
        at.append(ct[i+1])
        ax.append(cx[i+1])
        ay.append(cy[i+1])
        aex.append(cex[i+1])
        aey.append(cey[i+1])
        ott=ct[i+1]




fsx=[]
fsy=[]
fx=[]
fy=[]
ft=[]
for i in range (wdw,len(ct)):
    if i%wdw==0:
        mx=np.mean(cx[i-wdw:i])
        my=np.mean(cy[i-wdw:i])
        mt=np.mean(ct[i-wdw:i])
        sx=np.std(cx[i-wdw:i],ddof=1)
        sy=np.std(cy[i-wdw:i],ddof=1)
        fsx.append(sx)
        fsy.append(sy)
        fx.append(mx)
        fy.append(my)
        ft.append(mt)

fsx=np.array(fsx)
fsy=np.array(fsy)
plt.figure()
plt.plot(ct,cx,'.')
plt.plot(ct,xc,'-')
plt.plot(ft,fx,'-')
plt.plot(at,ax,'x')

plt.figure()
plt.plot(ct,cy,'.')
plt.plot(ct,yc,'-')
plt.plot(ft,fy,'-')
plt.plot(at,ay,'x')

plt.figure()
plt.plot(cx,cy,'.')
plt.plot(xc,yc,'-')
plt.plot(fx,fy,'-')
#plt.plot(ax,ay,'x')

plt.figure()
#plt.errorbar(cx,cy,ls='-.',yerr=cry,xerr=crx)
plt.errorbar(fx,fy,marker='.',ls='',yerr=fsy,xerr=fsx)
#plt.errorbar(cx,cy,ls='-',yerr=cey,xerr=cex)
#plt.plot(xc,yc,'-')
plt.plot(cx,cy,'x')

np.savetxt('D:\TAda\Code\Data\Da\d\Data TXY Selected %s.txt'
                 %namafile,np.c_[ft,fx,fy,fsx,fsy])
