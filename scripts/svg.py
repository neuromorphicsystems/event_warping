import logging
import pathlib
import subprocess
import sys


def fix(path: pathlib.Path):
    try:
        subprocess.run(
            [
                "svgcleaner",
                "--allow-bigger-file",
                "--no-defaults",
                str(path),
                str(path.with_name(f"{path.name}_fix.svg")),
            ],
            check=True,
            capture_output=True,
        )
        path.with_name(f"{path.name}_fix.svg").rename(path)
    except:
        logging.warning(sys.exc_info()[1])
