import argparse
from src.filters1D import get_coordinates_filters
from src.visualization import *

parser = argparse.ArgumentParser(description='Visualize 1D filters from Conv models')

parser.add_argument(
	'--models',
	help="path to models to consider",
	metavar='M',
	nargs='+',
    type=str,
    default='',
    required=True)

parser.add_argument(
	'--layers',
	help="index of layers in corresponding models",
	metavar='L',
	nargs='+',
    type=int,
    default='',
    required=True)

parser.add_argument(
	'--outdir',
	help="output directory",
    type=str,
    default='out')

parser.add_argument(
	'--title',
	help="title of the html page",
	type=str,
	default='Filter visualization')

args = parser.parse_args()

if len(args.models)!=len(args.layers):
	print('Number of layer indexes should be the same as number of models')
	exit()

coordinates, filters = get_coordinates_filters(args.models,args.layers)
generate_html(args.outdir,coordinates,filters,args.title)

