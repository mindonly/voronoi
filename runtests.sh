#!/usr/local/bin/zsh

Y_array=(768 900 1080)
X_array=(1024 1600 1920)

for i in {1..3}
do
    for seeds in 125 250 500
    do
        for j in {1..3} 
        do
            echo ----------------
            echo $seeds $X_array[j] $Y_array[j]
            echo ----------------
            ./voronoi $seeds $X_array[j] $Y_array[j]
        done
    done
done