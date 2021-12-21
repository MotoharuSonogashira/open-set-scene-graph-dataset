#!/bin/zsh

./plot_classes_vs_images.py -m infreq stat.pkl --dpi=300 -O infreq.pdf --figsize='(5,3)'
./plot_classes_vs_images.py -m rand stat.pkl --dpi=300 --figsize='(5,3)' -O rand.pdf
