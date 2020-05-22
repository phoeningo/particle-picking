# particle-picking
Install requirements:

pip install -r requirements.txt
#

Only two .py files are need for an automatic particle picking on tomogram ( already reconstruted from tilt series).

convert.py  is used to convert an input tomo into a little one, then excute pick_on_little.py to pick particles.

#Basic Usage:

step 1 :

  python convert.py --input [inputmrcname.mrc] --output [outputname.mrc]
  
step2:

  python pick_on_little.py --input [littlemrcname.mrc] --output [name.box] 

