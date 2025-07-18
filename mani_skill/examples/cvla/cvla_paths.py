from pathlib import Path


CVLA_DATASETS_PATH = Path("/work/dlclarge2/zhangj-zhangj-CFM/data/annotation")  # plase set this directory

# These are the update spoc models which are packaged as tar files
SPOC_ROOT_PATH = CVLA_DATASETS_PATH / "spoc" / "r2_dev"  # should contain annotations.json assets/
SPOC_DOWNLOAD_URL = "https://pub-2619544d52bd4f35927b08d301d2aba0.r2.dev/assets/{uid}.tar"
#SPOC_DOWNLOAD_URL = "<url to download /assets/{uid}.tar files>"
