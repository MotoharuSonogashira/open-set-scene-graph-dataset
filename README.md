# Open-Set Scene Graph Dataset

This is an open-set label preprocessor for the [Visual Genome](https://visualgenome.org) scene graph dataset.
The preprocessed dataset was used for open-set scene graph generation in the paper "[Towards Open-Set Scene Graph Generation with Unknown Objects](https://ieeexplore.ieee.org/abstract/document/9690166)" by Sonogashira et al. (IEEE Access, 2022). See the [main implementation repository](https://github.com/MotoharuSonogashira/open-set-scene-graph-generation) for the code of scene graph generation that uses the open-set dataset made by this repository.

## Installation

### Requirements
- Python 3.8

```bash
# clone this repository
git clone https://github.com/MotoharuSonogashira/open-set-scene-graph-dataset.git

# install dependencies
cd open-set-scene-graph-dataset
pip3 install --user -r requirements.txt
```

No build is required since preprocessing uses scripts in `data_tools` only. (The above-mentioned implementation repository also uses part of visualization scripts in `lib`.)

## Usage

The following command downloads the Visual Genome dataset and preprocesses it:
```bash
# continuing the previous section, assume that the current directory is the root of this repository
./download.bash
```
This produces the following files in the `data_tools` directory:
- `imdb_1024.h5`: image database file
- `VG-SGG-open.h5` and `VG-SGG-dicts-open.json`: open-set label database files 

## Acknowledgements

This repository is based on [danfeiX/scene-graph-TF-release](https://github.com/danfeiX/scene-graph-TF-release), the implementation of the paper "Scene Graph Generation by Iterative Message Passing" by Xu et al. (CVPR, 2017).


## Citations

```
@article{sonogashira2022towards,
  title={Towards Open-Set Scene Graph Generation with Unknown Objects},
  author={Sonogashira, Motoharu and Iiyama, Masaaki and Kawanishi, Yasutomo},
  journal={IEEE Access},
  volume={10},
  number={},
  pages={11574-11583},
  year={2022},
  publisher={IEEE}
}
```
