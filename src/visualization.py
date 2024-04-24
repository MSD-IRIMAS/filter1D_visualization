import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file, save, ColumnDataSource
from bs4 import BeautifulSoup
from .utils.utils import create_directory

list_colors = ['#332288', '#117733', '#999933', '#882255', '#CC3311', '#EE7733', '#88CCEE', '#44AA99', '#DDCC77', '#AA4499', '#CC6677', '#EECC66']

def generate_html(outdir, coordinates, filters, title):

	create_directory(outdir)
	save_filters_to_reduced_imgs(outdir,filters)
	plot_graph_to_html(outdir,coordinates,filters,title)
	adapt_html(outdir)


def save_filters_to_reduced_imgs(outdir, filters):

	dir_filter = outdir + '/static/filter_imgs/'
	create_directory(dir_filter)

	for f in range(len(filters)):
		model_dir = dir_filter + 'model_' + str(f) + '/'
		create_directory(model_dir)
		for i in range(filters[f].shape[0]):
			plt.plot(filters[f][i,:], linewidth=3.0, color=list_colors[f])
			plt.axis('off')
			plt.tight_layout()
			plt.savefig(model_dir + 'filter_' + str(i) + '.png')
			plt.clf()


def plot_graph_to_html(outdir, coordinates, filters, title):

	TOOLS="hover"
	TOOLTIPS = """
		<div>
			<img
	            src="@imgs" height="96" alt="@imgs" width="96"
	            style="float: center; margin: 0px 0px 0px 0px;"
	        ></img>
		</div>
	"""

	# set output to static HTML file
	output_file(filename=outdir+'/index.html', title=title)

	#create a source from filters of a model
	sources = []
	for f in range(len(filters)):
		source = ColumnDataSource(data=dict(
			x = coordinates[f][:,0],
			y = coordinates[f][:,1],
			imgs = ['static/filter_imgs/model_' + str(f) + '/filter_' + str(i) + '.png' for i in range(coordinates[f].shape[0])]
		))
		sources.append(source)

	# create a new plot with a title and axis labels
	p = figure(x_axis_label='x', y_axis_label='y', tools=TOOLS, tooltips=TOOLTIPS)
	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None
	p.xaxis.visible = False
	p.yaxis.visible = False

	# plot filters 2D as circles
	for s in range(len(sources)):
		circles = p.circle('x', 'y', size=10, color=list_colors[s], alpha=0.6, source=sources[s])

	# save
	save(p)


def adapt_html(outdir):

	# center the plot by modifying the html
	with open(outdir+'/index.html') as html_doc:
		soup = BeautifulSoup(html_doc, 'html.parser')
		olddiv = soup.find("div", {"class": "bk-root"})
	with open(outdir+'/index.html', 'wb') as html_doc:
		newdiv = olddiv
		newdiv['align']='center'
		olddiv.replace_with(newdiv)
		html_doc.write(soup.prettify("utf-8"))

