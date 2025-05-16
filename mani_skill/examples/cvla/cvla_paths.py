from pathlib import Path


CVLA_DATASETS_PATH = Path("<path to data directory>")  # plase set this directory

# These are the update spoc models which are packaged as tar files
SPOC_ROOT_PATH = CVLA_DATASETS_PATH / "spoc" / "r2_dev"  # should contain annotations.json assets/
SPOC_DOWNLOAD_URL = "<url to download /assets/{uid}.tar files>"
