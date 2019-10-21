import time
import sys
import numpy as np

def dp(N,A,B):
	R = np.dot(A,B)
	return R
    
length_of_array = int(sys.argv[1])
loops_reps = int(sys.argv[2])


x = np.ones(length_of_array,dtype=np.float32)
y = np.ones(length_of_array,dtype=np.float32)


finalans=0
timetaken=0

for i in range(1,loops_reps+1):
	startclock = time.monotonic()
	finalans=dp(length_of_array,x,y)
	endclock = time.monotonic()
	if i > loops_reps//2:
		timetaken+=endclock-startclock
        


print("Multiplication of two arrays:",finalans)

mult1 = 1000000
mult2 = 1000
secondaveragehalftime = timetaken/((loops_reps-(loops_reps//2)))
flopstime = (2*length_of_array)/(secondaveragehalftime)
bandwidth = length_of_array*8/(secondaveragehalftime*mult1*mult2)

print("N:",length_of_array," <T>:",secondaveragehalftime,"sec  B:",bandwidth,"GB/sec  F:",flopstime,"FLOP/sec")