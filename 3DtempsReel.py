import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################
print("GO") 

#FONCTIONS

def rechLignes(tab,y0):
    "Détermine les positions horizontales extrémales de l'objet détecté"
    L=[0,0]
    y=y0
    maxi=np.nansum(tab[0])
    while (L[1]-L[0])<seuil:
        L=[]
        while not(maxi>0):
            y+=1
            try:
                maxi=np.nansum(tab[y])
            except:
                return False
        L.append(y)
        while (maxi>0):
            y+=1
            try:
                maxi=np.nansum(tab[y])
            except:
                break
        L.append(y-1)
    
    if (L[1]-L[0])<seuil and 0:
        return (rechLignes(tab,y))
    
    return(L)
    

    
def rechColonnes(tab,L,x0):
    "Détermine les positions verticales extrémales de l'objet détecté"
    C=[0,0]
    x=x0
    maxi=np.nansum([tab[k][0] for k in range(L[0],L[1]+1)])
    while (C[1]-C[0])<seuil:
        C=[]
        while not(maxi>0):
            x+=1
            try:
                maxi=np.nansum([tab[k][x] for k in range(L[0],L[1]+1)])
            except:
                return False
        C.append(x)
        while (maxi>0):
            x+=1
            try:
                maxi=np.nansum([tab[k][x] for k in range(L[0],L[1]+1)])
            except:
                break
        C.append(x-1)
    
    if (C[1]-C[0])<seuil and 0:
        return (rechColonnes(tab,L,x))
        
    return(C)
    

def coef(sol,T):
    "Coefficient de régression quadratique R²"
    sse=sum( [ (T[i][1] - sol[0]*(T[i][0]**2) - sol[1]*T[i][0] - sol[2])**2 for i in range(len(T))] )
    Y=np.mean(T[:][1])
    sst=sum([ (T[i][1]-Y)**2 for i in range(len(T))])
    return((1-(sse/sst))**1)
    
    
def testPertu(tab):
    for i,x in enumerate(tab):
        for j,y in enumerate(tab[i+1:]):
            if abs(tab[i][0]-tab[j+i+1][0])<ecartMin or abs(tab[i][1]-tab[j+i+1][1])<ecartMin :
                print(tab)
                print(i,j+i+1)
                return True
    return False

    
###############################################################################
###############################################################################
###############################################################################

#CALIBRAGE

#Coordonnees des points dans l'espace (en cm)
coord=[ [52,175.8,0],[100.6,175.8,0],[54.9,94.2,0],[95,87.1,0.7],[279,175.8,0],[279,249,0],[279,182.7,-58.8],[278.3,83.7,-25.5]]
#Coordonnees aprés numérisation pas les caméras 0 et 1 (en pixel0)
#pixel0=[[102,165],[196,163],[102,327],[181,342],[565,153],[556,9],[523,172],[553,346]]
#pixel1=[[73,94],[212,127],[60,355],[184,379],[605,181],[600,25],[514,184],[572,367]]
#pixel0=[[117.5,206.6],[208,204],[111,369],[191,383.5],[577,193],[563.4,51],[538,208.5],[573,390]]
#pixel1=[[48,150],[190,174],[29,410],[162,434],[589,231],[581,77],[499,245],[560,448]]

pixel0=[[95,211],[188,209],[89,371],[170,387],[551,201],[538,58],[512,216],[546,391]]
pixel1=[[74,161],[216,180],[69,413],[195,435],[609,217],[595,62],[519,228],[587,439]]


B0,D0=[],[]
B1,D1=[],[]

for k in range( len(coord) ):
    B0.append([ coord[k][0],coord[k][1],coord[k][2],1,0,0,0,0,
        -coord[k][0]*pixel0[k][0],-coord[k][1]*pixel0[k][0],coord[k][2]*pixel0[k][0] ])
    B0.append([ 0,0,0,0,coord[k][0],coord[k][1],coord[k][2],1,
        -coord[k][0]*pixel0[k][1],-coord[k][1]*pixel0[k][1],coord[k][2]*pixel0[k][1] ])
    
    D0.append( [ -pixel0[k][0] ] )
    D0.append( [ -pixel0[k][1] ] )

    B1.append([ coord[k][0],coord[k][1],coord[k][2],1,0,0,0,0,
        -coord[k][0]*pixel1[k][0],-coord[k][1]*pixel1[k][0],coord[k][2]*pixel1[k][0] ])
    B1.append([ 0,0,0,0,coord[k][0],coord[k][1],coord[k][2],1,
        -coord[k][0]*pixel1[k][1],-coord[k][1]*pixel1[k][1],coord[k][2]*pixel1[k][1] ])
    
    D1.append( [ -pixel1[k][0] ] )
    D1.append( [ -pixel1[k][1] ] )
    
Delta0= - np.dot( np.dot( np.linalg.inv( np.dot( np.transpose(B0) , B0 ) ) , np.transpose(B0) ) , D0)
Delta1= - np.dot( np.dot( np.linalg.inv( np.dot( np.transpose(B1) , B1 ) ) , np.transpose(B1) ) , D1)


###############################################################################
###############################################################################
###############################################################################
  
#VARIABLES
    
seuil=3
filtrage=20
selectFrame=100/100
ecartMin=2

#Booleens
PERTURB=0
#CAMERA=1
TEST=1
BOUCLE=0
affPhotos=1
reverseGraphe=0
objMin=1000

###############################################################################
###############################################################################
###############################################################################

#CODE

try:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
except:
    abc=1    
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cv2.CV_CAP_PROP_EXPOSURE=0
#cv2.VideoCapture(CV_CAP_PROP_EXPOSURE=0.1)
#PAUSE
t0=time()
while time()-t0<5:
    abc=1

#CREATION DE L'ETALON (avec 20 images)
ret, frame00 = cap0.read()
ret, frame01 = cap1.read()

#plt.imshow(frame00)

gray00 = cv2.cvtColor(frame00, cv2.COLOR_BGR2GRAY)
gray01 = cv2.cvtColor(frame01, cv2.COLOR_BGR2GRAY)

for k in range(1,30):
    t0=time()
    print(30-k)
    while time()-t0<1/20:
        abc=1
    ret, frame00 = cap0.read()
    ret, frame01 = cap1.read()
    gray10 = cv2.cvtColor(frame00, cv2.COLOR_BGR2GRAY)
    gray00 = ( gray10 +np.ones((480,640)) +gray00 )
    
    gray11 = cv2.cvtColor(frame01, cv2.COLOR_BGR2GRAY)
    gray01 = ( gray11 +np.ones((480,640)) +gray01 )
    
grayI0=(1/30)*np.array(gray00)  #ETALON CAM 0
grayI1=(1/30)*np.array(gray01)  #ETALON CAM 1

"""plt.subplot(121)
plt.imshow(frame00)
plt.subplot(122)
plt.imshow(frame01)"""


#INITALISATIONS

acq=0           #Nombre d'images depuis la détection du projectile
    
tabCoord0=[]     #Tableau des coordonnees du projectile
PHOTOS0=[]       #Liste des images du projectile
tabIm0=[]        #Liste des images du projectile filtrees

tabCoord1=[]    
PHOTOS1=[]      
tabIm1=[] 

delai0,delai1=0.15,0.43
T0,T1=[],[]
T=[]
TT=[]
TM=[]
t2=0
ti=0
XYZ=[[],[],[]]

t=0
T=[]
T0=[]
n=1
impact=[]
XYZ=[[],[],[]]
tir=[]
tir2=[]

vAM=10 #m/s

while 1:
    t=time()
    
    #print("lel",t-t2)
    
    ret, frame1 = cap1.read()
    t1=time()    
    
    ret, frame0 = cap0.read()
    t2=time()   


    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)  
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  
    
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)
    dif0=np.sqrt(np.abs(gray0+np.zeros((480,640)) -(grayI0))-filtrage*np.ones((480,640)))
    dif1=np.sqrt(np.abs(gray1+np.zeros((480,640)) -(grayI1))-filtrage*np.ones((480,640)))

    
    testObj0=np.nansum([np.nansum(k) for k in dif0 ]) 
    testObj1=np.nansum([np.nansum(k) for k in dif1 ]) 


    if acq>0:
        print("acq : ",acq)
        print(t1-t)
        print(t2-t1)
        acq+=1
        
        PHOTOS0.append(frame0)
        PHOTOS1.append(frame1)
        
        
        T.append( int((t-ti)*1000)/1000 ) 
        
        L0=rechLignes(dif0,0)
        if L0:
            C0=rechColonnes(dif0,L0,0)
            if not C0:
                break
        else:
            break
      
        
        L1=rechLignes(dif1,0)
        if L1:
            C1=rechColonnes(dif1,L1,0)
            if not C1:
                break
        else:
            break

        [x0,y0],[x1,y1]=[np.mean(C0),np.mean(L0)],[np.mean(C1),np.mean(L1)]
        
        tabCoord0.append([x0,y0])
        tabCoord1.append([x1,y1])
        tabIm0.append(dif0)
        tabIm1.append(dif1) 
        
        B=[ [ x0*Delta0[8][0]-Delta0[0][0],x0*Delta0[9][0]-Delta0[1][0],x0*Delta0[10][0]-Delta0[2][0] ],
            [ y0*Delta0[8][0]-Delta0[4][0],y0*Delta0[9][0]-Delta0[5][0],y0*Delta0[10][0]-Delta0[6][0] ],
            [ x1*Delta1[8][0]-Delta1[0][0],x1*Delta1[9][0]-Delta1[1][0],x1*Delta1[10][0]-Delta1[2][0] ],
            [ y1*Delta1[8][0]-Delta1[4][0],y1*Delta1[9][0]-Delta1[5][0],y1*Delta1[10][0]-Delta1[6][0] ] ]
            
        C=[ [x0-Delta0[3][0]],[y0-Delta0[7][0]],[x1-Delta1[3][0]],[y1-Delta1[7][0]]]
        
        xyz=( - np.dot( np.dot( np.linalg.inv( np.dot( np.transpose(B) , B ) ) , np.transpose(B) ) , C) )
        XYZ[0].append(int(xyz[0][0]))
        XYZ[1].append(int(xyz[1][0]))
        XYZ[2].append(int(xyz[2][0]))
    
        
       
            

    elif testObj0>objMin and testObj1>objMin and acq==0 and not BOUCLE:

        L0=rechLignes(dif0,0)
        if L0:
            C0=rechColonnes(dif0,L0,0)
            if not C0:
                L0=[0,0]
                C0=[0,0]
        else:
            L0=[0,0]
            C0=[0,0]

        L1=rechLignes(dif1,0)
        if L1:
            C1=rechColonnes(dif1,L1,0)
            if not C1:
                L1=[0,0]
                C1=[0,0]
        else:
            L1=[0,0]
            C1=[0,0]

        if not( (C0[0]*L0[0]==0 or L0[1]==479 or C0[1]==639) or (C1[0]*L1[0]==0 or L1[1]==479 or C1[1]==639) ):
            [x0,y0],[x1,y1]=[np.mean(C0),np.mean(L0)],[np.mean(C1),np.mean(L1)]
            
            tabCoord0.append([x0,y0])
            tabCoord1.append([x1,y1])
            tabIm0.append(dif0)
            tabIm1.append(dif1) 
            
            B=[ [ x0*Delta0[8][0]-Delta0[0][0],x0*Delta0[9][0]-Delta0[1][0],x0*Delta0[10][0]-Delta0[2][0] ],
                [ y0*Delta0[8][0]-Delta0[4][0],y0*Delta0[9][0]-Delta0[5][0],y0*Delta0[10][0]-Delta0[6][0] ],
                [ x1*Delta1[8][0]-Delta1[0][0],x1*Delta1[9][0]-Delta1[1][0],x1*Delta1[10][0]-Delta1[2][0] ],
                [ y1*Delta1[8][0]-Delta1[4][0],y1*Delta1[9][0]-Delta1[5][0],y1*Delta1[10][0]-Delta1[6][0] ] ]
                
            C=[ [x0-Delta0[3][0]],[y0-Delta0[7][0]],[x1-Delta1[3][0]],[y1-Delta1[7][0]]]
            
            xyz=( - np.dot( np.dot( np.linalg.inv( np.dot( np.transpose(B) , B ) ) , np.transpose(B) ) , C) )
            XYZ[0].append(int(xyz[0][0]))
            XYZ[1].append(int(xyz[1][0]))
            XYZ[2].append(int(xyz[2][0]))
            
            PHOTOS0.append(frame0)
            PHOTOS1.append(frame1)
        
            acq=1
            
            ti=t

            T.append(0)
            
            
    if 1 and acq>4:
        [aX,bX]=np.polyfit(T,XYZ[0],1)
        [aZ,bZ]=np.polyfit(T,XYZ[2],1)
        [aY,bY,cY]=np.polyfit(T,XYZ[1],2)
        
        t0=-bX/aX
        [Y,Z]=[aY*t0*t0 + bY*t0 + cY , aZ*t0 + bZ]
        T0.append( int(t0*1000)/1000 )
        impact.append([int(Y),int(Z)])

        angle=np.arctan(Y/Z)*180/np.pi
        d=(Y*Y + Z*Z)**(1/2)/100
        tAM=d/vAM
        
        tir.append(int(angle))
        tir2.append( int(tAM*1000)/1000 )
        
        if t-ti>t0-tAM-0.06 and 1:
            print("STOP", t,t0-tAM)
            #print("AAA")
            #print(t,t0,tAM)
            break
        
        
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print("")
print("Angle prevu: ",tir)
print("Duree tir/impact: ",tir2)
print("Lieu d'impact prévu: ",impact)
print("Temps d'impact prévu: ",T0)
print("Temps d'acquisition: ",T)
print("")
print("tir a t= : ",int((t0-tAM)*1000)/1000," angle: ",int(angle))

t=np.linspace(0,t0,1000)

plt.subplot(311)
x=[aX*k+bX for k in t]
plt.plot(t,x)

plt.subplot(312)
z=[aZ*k+bZ for k in t]
plt.plot(t,z)

plt.subplot(313)
y=[aY*k*k+bY*k+cY for k in t]
plt.plot(t,y)

plt.show()
    
cap0.release()
cap1.release()
cv2.destroyAllWindows()
    
XYZ=[[],[],[]]
for k in range(len(tabCoord0)):
    [x0,y0],[x1,y1]=tabCoord0[k],tabCoord1[k]
    B=[ [ x0*Delta0[8][0]-Delta0[0][0],x0*Delta0[9][0]-Delta0[1][0],x0*Delta0[10][0]-Delta0[2][0] ],
        [ y0*Delta0[8][0]-Delta0[4][0],y0*Delta0[9][0]-Delta0[5][0],y0*Delta0[10][0]-Delta0[6][0] ],
        [ x1*Delta1[8][0]-Delta1[0][0],x1*Delta1[9][0]-Delta1[1][0],x1*Delta1[10][0]-Delta1[2][0] ],
        [ y1*Delta1[8][0]-Delta1[4][0],y1*Delta1[9][0]-Delta1[5][0],y1*Delta1[10][0]-Delta1[6][0] ] ]
        
    C=[ [x0-Delta0[3][0]],[y0-Delta0[7][0]],[x1-Delta1[3][0]],[y1-Delta1[7][0]]]
    
    xyz=( - np.dot( np.dot( np.linalg.inv( np.dot( np.transpose(B) , B ) ) , np.transpose(B) ) , C) )
    XYZ[0].append(int(xyz[0][0]))
    XYZ[1].append(int(xyz[1][0]))
    XYZ[2].append(int(xyz[2][0]))
"""print(XYZ)

print(tabCoord0)
print(tabCoord1)

print("-----")
#print(T0)
#print(T1)

print(T)
print(TT)
print(TM)"""














#print("rac")
solXY=np.polyfit(XYZ[0],XYZ[1],2)
solZY=np.polyfit(XYZ[2],XYZ[1],2)


c0=solXY[2]
[a,b,c]=solZY
d=(b*b-4*a*(c-c0))**(1/2)
#print("impact: ",c0,[(-b+d)/(2*a) , (-b-d)/(2*a)] )






"""
[a,b,c]=solXY
d=(b*b-4*a*c)**(1/2)
print( (-b+d)/(2*a) , (-b-d)/(2*a) )
[a,b,c]=solZY
d=(b*b-4*a*c)**(1/2)
print( (-b+d)/(2*a) , (-b-d)/(2*a) )


plt.subplot(221)
x=[k for k in range(0,300)]
yx=[solXY[0]*(k**2)+solXY[1]*k+solXY[2] for k in x]
plt.plot(x,yx)
plt.xlim(300,0)


plt.subplot(222)
z=[k for k in range(0,200)]
yz=[solZY[0]*(k**2)+solZY[1]*k+solZY[2] for k in z]
plt.plot(z,yz)
plt.xlim(200,0)

plt.show()"""

#cv2.imshow("Etalon",frame01)
#cv2.imshow("Image",PHOTOS1[3])
#cv2.imshow("Aprés filtrage", np.sqrt(np.abs( cv2.cvtColor(PHOTOS1[3], cv2.COLOR_BGR2GRAY)+np.zeros((480,640)) -(grayI1))-filtrage*np.ones((480,640))))


if affPhotos:
    for k,im in enumerate(PHOTOS0):
        print(k)
        if k<20:
            cv2.imshow(str(k),im)
    for k,im in enumerate(PHOTOS1):
        print(k)
        if k<20:
            cv2.imshow(str(k)+"100",im)
            
while affPhotos or 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
    

