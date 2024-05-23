import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from bokeh.plotting import figure, output_file, save, ColumnDataSource, show

from .utils.utils import create_directory, generate_distinct_colors


def generate_html(outdir: str,
                  coordinates: list,
                  filters: list,
                  list_labels:list,
                  list_colors:list,
                  title: str):
    """Function to generate the html file of the plot.

    Parameters
    ----------
    outdir: str
		The output directory to store the results.
    coordinates: list
		A list containing the 2D filters coordinates of each model
		of the chosen layer.
    filters: list
		A list containing the original filters of each model of the
        chosen layer.
    list_labels: list
        A list of labels for each model's filters.
    title: str
		The title of the figure produced.
    """
    create_directory(outdir)
    list_colors = _save_filters_to_reduced_imgs(outdir, filters, list_colors)
    _plot_graph_to_html(outdir, coordinates, filters, title, list_colors, list_labels)
    _adapt_html(outdir)


def _save_filters_to_reduced_imgs(outdir, filters, list_colors):

    if list_colors is None:
        list_colors = generate_distinct_colors(num_colors=len(filters))
    else:
        list_colors = list_colors

    dir_filter = outdir + "/static/filter_imgs/"
    create_directory(dir_filter)

    for f in range(len(filters)):
        model_dir = dir_filter + "model_" + str(f) + "/"
        create_directory(model_dir)
        for i in range(filters[f].shape[0]):
            plt.plot(filters[f][i, :], linewidth=3.0, color=list_colors[f])
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(model_dir + "filter_" + str(i) + ".png")
            plt.clf()

    return list_colors


def _plot_graph_to_html(outdir, coordinates, filters, title, list_colors, list_labels):

    TOOLS = "hover"
    TOOLTIPS = """
		<div>
			<img
	            src="@imgs" height="96" alt="@imgs" width="96"
	            style="float: center; margin: 0px 0px 0px 0px;"
	        ></img>
		</div>
	"""

    # set output to static HTML file
    output_file(filename=outdir + "/index.html", title=title)

    # create a source from filters of a model
    sources = []
    for f in range(len(filters)):
        source = ColumnDataSource(
            data=dict(
                x=coordinates[f][:, 0],
                y=coordinates[f][:, 1],
                imgs=[
                    "static/filter_imgs/model_" + str(f) + "/filter_" + str(i) + ".png"
                    for i in range(coordinates[f].shape[0])
                ],
            )
        )

        sources.append(source)

    # create a new plot with a title and axis labels
    p = figure(x_axis_label="x", y_axis_label="y", tools=TOOLS, tooltips=TOOLTIPS)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

    # plot filters 2D as circles
    for s in range(len(sources)):

        circles = p.circle(
            "x", "y", size=10, color=list_colors[s], alpha=0.6, source=sources[s], 
            legend_label=list_labels[s]
        )

    p.legend.location = "top_right"

    # save
    save(p)


def _adapt_html(outdir):

    # center the plot by modifying the html
    with open(outdir + "/index.html") as html_doc:
        soup = BeautifulSoup(html_doc, "html.parser")
        body_tag = soup.find("body")
        body_tag["style"] = (
            "display: flex; justify-content: center; align-items: center;"
        )
    with open(outdir + "/index.html", "wb") as html_doc:
        html_doc.write(soup.prettify("utf-8"))
