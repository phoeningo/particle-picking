import os
import mrcfile
import sys
import argparse
import numpy as np
from numba import cuda,jit
import time
import math
from skimage import measure,exposure,morphology

parser=argparse.ArgumentParser(description='T')
parser.add_argument('--good_star',type=str)
parser.add_argument('--bad_star',type=str)
parser.add_argument("--input",type=str)
parser.add_argument("--output",type=str,default='')
parser.add_argument("--k_size",type=int,default=1)
parser.add_argument("--w_size",type=int,default=3)
parser.add_argument("--thre",type=int,default=0)
parser.add_argument("--min_size",type=int,default=30)
parser.add_argument("--max_size",type=int,default=150)
parser.add_argument("--star",type=str,default='')
parser.add_argument('--number',type=int,default=350)
parser.add_argument("--tick",type=int,default=5)
parser.add_argument('--shrink',type=float,default=1)
parser.add_argument("--flo",type=float,default=0.05)
parser.add_argument("--level",type=float,default=0.5)
parser.add_argument("--radius",type=int,default=2)
parser.add_argument("--mode",type=str,default='little')
parser.add_argument('--is_little',type=bool,default=False)
parser.add_argument('--min_distance',type=int,default=100)
parser.add_argument('--max_distance',type=int,default=300)
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--grow',type=int,default=3)
parser.add_argument('--ops',type=int,default=1)
parser.add_argument('--iter_num',type=int,default=4)
parser.add_argument('--tag',type=str,default='C')
args=parser.parse_args()


@cuda.jit
def bw(I,threshold,O):
    i,j,k=cuda.grid(3)
    if I[i,j,k]>threshold:
        O[i,j,k]=1
    else:
        O[i,j,k]=0



def dis(a,b):
    return np.sqrt(np.sum((np.array(a)-np.array(b))*(np.array(a)-np.array(b))))

def list_mean(a,b):
    return list((np.array(a)+np.array(b))/2)

def insert(arr,x,t):
    for item in arr:
        if dis(item,x)<t:
         #   arr[arr.index(item)]=list_mean(item,x)
            return
    arr.append(x)  

def combine(all,temp,t):
    for x in temp:
        insert(all,x,t)

os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu) 

t1=time.time()
inp=mrcfile.mmap(args.input)
D=np.uint8(255*np.float32(inp.data))
inp=0

#-----------------------
final_coord=[]
thre=254
grow=0
tmp=np.zeros(shape=D.shape,dtype=np.float32)
tick=args.tick

#tmp=morphology.opening(D,morphology.ball(args.radius))
scale_size=2*args.k_size+1
cs=scale_size*2
bs=args.k_size
z,y,x=D.shape
griddim=z,y,x
#print (x,y,z)
blockdim=1
dz,dy,dx=z,y,x

it=1
tick=args.tick
max_size=args.max_size
min_size=args.min_size
#-----------------------
while it<args.iter_num:
    coord=[]
    #areas=[]
    last=len(final_coord)
   # thre=190
    bw[griddim,blockdim](D,np.uint8(thre),tmp)
    label=measure.label(tmp)                       
    ps=measure.regionprops(label)
   # print('current len: '+str(len(ps)))
    for item in ps:
        if (args.min_size+it*args.grow)/cs < item.equivalent_diameter and item.equivalent_diameter < (args.max_size+it*args.grow)/cs:
            zz,ty,tx=item.centroid
            coord.append(item.centroid)

    combine(final_coord,coord,args.min_distance/(2*cs))
    current=len(final_coord)
    grow=current-last
    print ('find ',grow,' more particles.')
    print ('now picked ',len(final_coord))
    thre=thre-tick
    it=it+1

#-------------------------------------------------------------------------
    if args.output=='':
        filename=args.input.split('/')
        mrcname=filename[len(filename)-1]
        nums=mrcname.split('.')[0] 
        outfile_name=nums+args.tag+'.coords'
        f=open(outfile_name,'w')
        print('write file :'+outfile_name)
    else:
        f=open(args.output,'w')
    for item in final_coord:
        (x,y,z)=item
        sr=""
        if args.mode=='pick':
            sr=str(z*cs+bs)+" "+str(y*cs+bs)+" "+str(x*cs+bs)+"\n"
        else:
            sr=str(z+bs)+" "+str(y+bs)+" "+str(x+bs)+"\n"
        f.write(sr)
    f.close()
print('time cost:'+str(time.time()-t1))
