import os
import mrcfile
import sys
import argparse
import numpy as np
from numba import cuda,jit
import time
import math
from skimage import measure,exposure

parser=argparse.ArgumentParser(description='T')
parser.add_argument("--input",type=str)
parser.add_argument("--output",type=str,default='')
parser.add_argument("--k_size",type=int,default=1)
parser.add_argument("--w_size",type=int,default=3)
parser.add_argument("--thre",type=int,default=0)
parser.add_argument("--box_size",type=int,default=100)

parser.add_argument("--tick",type=int,default=5)
parser.add_argument("--flo",type=float,default=0.05)
parser.add_argument("--level",type=float,default=0.1)
parser.add_argument("--radius",type=int,default=2)
parser.add_argument("--mode",type=str,default='convert')

args=parser.parse_args()


#-----------------------------------------------------
@cuda.jit
def bw(I,threshold,O):
    i,j,k=cuda.grid(3)
    if I[i,j,k]>threshold:
        O[i,j,k]=255
    else:
        O[i,j,k]=0

@cuda.jit
def scale(arr,cs,allcs,tmp,w_size):
    i,j,k=cuda.grid(3)
    t=arr[i*cs:i*cs+cs,j*cs:j*cs+cs,k*cs:k*cs+cs]
    x,y,z=t.shape
    m=0
    for xi in range(x):
        for yj in range(y):
            for zk in range(z):
                m=m+float(t[xi,yj,zk])/(x*y*z)
    tmp[i+w_size,j+w_size,k+w_size]=m
    
@cuda.jit
def mmean(arr,tmp,wcs):
    i,j,k=cuda.grid(3)
    t=arr[i:i+wcs,j:j+wcs,k:k+wcs]
    x,y,z=t.shape
    m=0
    for xi in range(x):
        for yj in range(y):
            for zk in range(z):
                m=m+float(t[xi,yj,zk])/(x*y*z)
    tmp[i,j,k]=m

@cuda.jit
def filt(img,tem,omin,omax):
    i,j,k=cuda.grid(3)
    tem[i,j,k]=np.uint8(255*(float(img[i,j,k])-omin)/(omax-omin))


def dis(a,b):
    return np.sqrt(np.sum((np.array(a)-np.array(b))*(np.array(a)-np.array(b))))

def list_mean(a,b):
    return list((np.array(a)+np.array(b))/2)

def insert(arr,x,t):
    for item in arr:
        if dis(item,x)<t:
            arr[arr.index(item)]=list_mean(item,x)
            return
    arr.append(x)  

def combine(all,temp,t):
    for x in temp:
        insert(all,x,t)

#---------------------------------------------------------------------

t1=time.time()
if args.mode!='convert' :
    print('strongly recommend you to set mode as convert (which is the default value) at the current version ')
    quit()

#print(args.input.split('.')[0])
#
i=mrcfile.mmap(args.input,'r')
img=np.zeros(i.data.shape,dtype=np.uint8)

z,y,x=i.data.shape

bx=int(x/2)
by=int(y/2)
bz=int(z/2)

gt=np.arange(256,dtype=np.int)
for xi in range(128):
    gt[128+xi]=xi-128
gt=np.int8(gt)


imax=0
imin=0
ra=0
print ('calc max&min...')
for zi in range(bz):
    if zi%50==0 :
        i.close()
        i=0
        i=mrcfile.mmap(args.input,'r')
    ma=np.max(i.data[zi*2])
    mi=np.min(i.data[zi*2])
    if ma>imax:
        imax=ma
    if mi<imin:
        imin=mi

print ('converting...')

ra=0
for zi in range(bz*2):
    if zi%50==0 :
        i.close()
        i=0
        i=mrcfile.mmap(args.input,'r')
    img[zi]=np.uint8(255)-np.uint8(255*(i.data[zi]-imin)/(imax-imin))

i.close()
i=0


#----------------------------------------------------------
print ('dividing...')
#--------------------------

cs=args.k_size*2+1
sz=args.k_size

wcs=args.w_size*2+1
wsz=args.w_size
allcs=cs*wcs
allsz=(allcs-1)/2

sx=int(bx/cs)
sy=int(by/cs)
sz=int(bz/cs)


avtp=np.zeros([sz,sy,sx],dtype=np.float32)

for i in range(8):
    print ('sub '+str(i+1))
    tp=np.zeros([bz,by,bx],dtype=np.uint8)
    
    if i%4==0:
        tempz=np.zeros([bz,y,x],dtype=np.uint8)
        for bzi in range(bz):
            tempz[bzi]=img[bzi*2+int(i/4)%2]
    if i%2==0:
        tempy=np.zeros([bz,by,x],dtype=np.uint8)
        for byi in range(by):
            tempy[:,byi]=tempz[:,byi*2+int(i/2)%2]
    for bxi in range(bx):
        tp[:,:,bxi]=tempy[:,:,bxi*2+i%2]


    tmp=np.zeros([sz+wcs-1,sy+wcs-1,sx+wcs-1],dtype=np.float32)

#----------------------------------------------------------------------
    griddim=sz,sy,sx

    blockdim=1


    scale[griddim,blockdim](tp,cs,allcs,tmp,wsz)   

    ttmp=np.zeros([sz,sy,sx],dtype=np.float32)
    top=np.max(tmp)
    m_mean=np.mean(tmp)

    tmp=exposure.equalize_hist(tmp)

    mmean[griddim,blockdim](tmp,ttmp,wcs)
    ttmp=np.float32(exposure.equalize_hist(ttmp))

    #print (np.max(ttmp),np.min(ttmp),np.mean(ttmp))

    tmp=0
    avtp=avtp+ttmp/8
    ttmp=0

#--------------------------------------------------------------------
print('Data conversion completed.')
if args.mode=='convert':
    print('writing...') 
    o=mrcfile.new_mmap(args.output,overwrite='True',shape=avtp.shape,mrc_mode=0)
    o.set_data(avtp)
    o.flush()
    o.close()
    
    

print ('time cost:',time.time()-t1)








