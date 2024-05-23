import hydra
from omegaconf import DictConfig, OmegaConf

from src.filters1D import get_coordinates_filters
from src.visualization import generate_html


@hydra.main(config_name="config_hydra.yaml", config_path="config")
def main(args: DictConfig):

    # save configuration file of the experiment used
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    # assert the number of models is the same as the number of layer indices
    assert len(args.model_paths) == len(args.layer_indices) == len(args.list_labels) == len(args.list_colors)

    coordinates, filters = get_coordinates_filters(
        model_paths=args.model_paths,
        layer_indices=args.layer_indices,
        distance=args.distance,
    )
    
    generate_html(
        outdir=args.output_directory,
        coordinates=coordinates,
        filters=filters,
        list_labels=args.list_labels,
        list_colors=args.list_colors,
        title=args.title,
    )


if __name__ == "__main__":
    main()
