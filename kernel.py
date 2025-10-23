# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -

import nbformat as nbf
from pathlib import Path

KERNEL_NAME = "ml-tech-assignment"
DISPLAY_NAME = "ML Tech Assignment"


for nb_path in Path("notebooks").glob("**/*.ipynb"):
    nb = nbf.read(nb_path, as_version=nbf.NO_CONVERT)
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata.kernelspec["name"] = KERNEL_NAME
    nb.metadata.kernelspec["display_name"] = DISPLAY_NAME
    nbf.write(nb, nb_path)
    print("Updated:", nb_path)
