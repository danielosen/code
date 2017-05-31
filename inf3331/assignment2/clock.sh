#!/bin/bash
while sleep 1; do
string=$("date")
arg=$1
flag=0
thisstring="--apm"
if [ "$arg" == "$thisstring" ]; then
	flag=1
fi
counter=0
for i in $string; do
	counter=$(($counter+1))
	if (( $counter >= 4 )); then
		systemtime=$i;
	fi
done
hours=${systemtime:0:2}
minutes=${systemtime:3:2}
seconds=${systemtime:6:2}
if [ $flag -eq 1 ]; then
	#am pm system is weird, 00 -> 12 am, 12 -> 12 pm
	if (( $hours > 12 )); then
		echo "$(($hours%12+!$hours*12)):$minutes PM"
	else
		echo "$(($hours%12+!$hours*12)):$minutes AM"
	fi
else
	echo $hours:$minutes
fi
done
#while sleep 1;do
#	string = date
#	done &
