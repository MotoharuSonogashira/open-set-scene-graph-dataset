#!/bin/bash

# download the Visual Genome dataset (images, metadata, and labels)
cd data_tools/VG
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
wget http://svl.stanford.edu/projects/scene-graph/VG/image_data.json
wget http://svl.stanford.edu/projects/scene-graph/VG/VG-scene-graph.zip

# decompress and rearrange them
unzip -j images.zip -d images
unzip -j images2.zip -d images
unzip -j VG-scene-graph.zip -d .

# create an image database
cd .. # go to data_tools
./create_imdb.sh

# create a open-set label database using the image database
./create_roidb_open.sh
