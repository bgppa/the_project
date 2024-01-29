#!/bin/bash
filename="binclf.py"

for i in {0..10..1}
do
	for n in "SEG" "SIN" "IMP"
	do
		echo "Running: python3 ${filename} ${i} ${n}"
		cmd=`python3 ${filename} ${i} ${n}`
	done
done
echo "*** DONE ***"
