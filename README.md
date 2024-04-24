# filter1D_visualization

Automatically generate a html (`index.html`) page showing a 2D space of 1D convolution-based filters as usually used in [Deep Learning for Time Series Classification](https://msd-irimas.github.io/pages/dl4tsc/).


## Usage

```
python main.py --models M [M ...] --layers L [L ...] [--outdir OUTDIR] [--title TITLE]
```
With:
- `M`: a list of Deep model paths
- `L`: a list of layer indexes in corresponding models
- `OUTDIR`: output directory to generate the html file (default: `out/`)
- `TITLE`: title of the html page (default: 'Filter visualization')


## Example

```
python main.py --models example/lite_car_0.hdf5 example/lite_car_1.hdf5 --layers -5 -5
```
This generate [this html file](https://maxime-devanne.com/pages/filter1D_visualization/), visualizing filters from the last conv layer of two different [Lite classifiers](https://github.com/MSD-IRIMAS/LITE/) trained on Car dataset.
