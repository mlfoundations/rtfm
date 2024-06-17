import fire

from rtfm.data_sources.utils import generate_files_from_csv


def main(csv: str, out_dir: str, target_colname: str, to_regression: bool):
    generate_files_from_csv(csv, out_dir, to_regression=to_regression)
    return


if __name__ == "__main__":
    fire.Fire(main)
