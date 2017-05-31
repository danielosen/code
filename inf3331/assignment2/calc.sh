#!/bin/bash
#sum,product,max,min
#first parameter specifies operation
if (( $# > 1 )); then
case "$1" in
	S)
		shift
		args=0
		for i in $@; do
			args=$(($args+$i))
		done
		;;
	P)
		shift
		args=1
		for i in $@; do
			args=$(($args*$i))
		done
		;;
	M)
		shift
		args=$1
		shift
		for i in $@; do
				echo start: $args
			if (( $i > $args )); then
				args=$i
				echo verdi: $args
			fi
		done
		;;
	m)
		shift
		args=$1
		shift
		for i in $@; do
			if (( $args > $i )); then
				args=$i
			fi
		done
esac
fi
echo $args
#lol="1+1"
#echo $[lol]

#for i in $@; do
#	#sum: start:args=0 args=$[args+$i]
#	#mult start:args=1 args=$[$args*$i]
#	echo $args
#done