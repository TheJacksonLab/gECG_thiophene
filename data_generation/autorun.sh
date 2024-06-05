#!/bin/sh
cd tp_gen
for tpi in $(seq 0 199); do
    if [ -d "tp$tpi" ]; then
        cd "tp$tpi"
	echo $tpi
	for i in $(seq 0 100); do
            if [ -d "str_$i" ]; then
                cd "str_$i"
                orca lstr.inp > log
                rm *gbw *densities *_property.txt
                cd ../
            fi
        done
        cd ../
    fi
done
cd ..

