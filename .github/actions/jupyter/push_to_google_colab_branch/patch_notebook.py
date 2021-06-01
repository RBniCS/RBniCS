# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Script to patch notebook to:
1) add FEniCS installation cell
2) add RBniCS installation cell
3) download auxiliary files (e.g. meshes)
4) update images to point to github raw and links to point to colab
"""

import glob
import io
import os
import sys
import nbformat

# Get notebook name
assert len(sys.argv) == 2
filename = sys.argv[1]
dirname = os.path.dirname(filename)
basename = os.path.basename(filename)
top_dir = filename.split(os.sep)[0]

# Get cell insertion location based on basename
if basename.startswith("tutorial"):
    insert_base = 1
elif basename.startswith("generate_mesh"):
    insert_base = 0
else:
    raise RuntimeError("Invalid notebook")

# Read in notebook content
with io.open(filename, "r", encoding="utf8") as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

# 1. Add FEniCS installation cell
fenics_installation_cell = """# Install FEniCS
try:
    import dolfin
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/fenics-install.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"
    import dolfin"""  # noqa: E501
fenics_installation_cell = nbformat.v4.new_code_cell(fenics_installation_cell)
fenics_installation_cell.id = "fenics_installation"
nb.cells.insert(insert_base, fenics_installation_cell)

# 2. Add RBniCS installation cell
rbnics_installation_cell = """# Install RBniCS
try:
    import rbnics
except ImportError:
    !pip3 install git+https://github.com/RBniCS/RBniCS.git
    import rbnics
import rbnics.utils.config
assert "dolfin" in rbnics.utils.config.config.get("backends", "required backends")"""
rbnics_installation_cell = nbformat.v4.new_code_cell(rbnics_installation_cell)
rbnics_installation_cell.id = "rbnics_installation"
nb.cells.insert(insert_base + 1, rbnics_installation_cell)

# 3. Add cell to download auxiliary files (e.g. meshes)
aux_urls = dict()

if basename.startswith("tutorial"):
    aux_create_dirs = set()
    aux_download_files = set()
    for aux_file in glob.glob(os.path.join(dirname, "**", "*"), recursive=True):
        aux_real_file = aux_file
        while os.path.islink(aux_real_file):
            aux_real_file = os.path.join(
                top_dir, os.path.relpath(os.path.realpath(aux_real_file), os.path.realpath(top_dir)))
        if os.path.isdir(aux_real_file):
            continue
        else:
            assert os.path.isfile(aux_real_file), (
                f"While processing {filename}, {aux_real_file} is not a file: ("
                + f"exists: {os.path.exists(aux_real_file)}, "
                + f"islink: {os.path.islink(aux_real_file)}, "
                + f"isdir: {os.path.isdir(aux_real_file)})")
            aux_url = (
                f"https://github.com/RBniCS/RBniCS/raw/{os.getenv('BRANCH', 'master')}"
                + f"/{aux_real_file.replace('patched_', '')}")
        _, aux_file_ext = os.path.splitext(aux_file)
        if aux_file_ext not in (".ipynb", ".link"):
            aux_urls[os.path.relpath(aux_file, top_dir)] = aux_url
            aux_urls[os.path.relpath(aux_file, dirname)] = aux_url
        if aux_file_ext not in (".ipynb", ".link", ".png"):
            aux_dir = os.path.relpath(os.path.dirname(aux_file), dirname)
            if aux_dir not in ("", "."):
                aux_create_dirs.add(f"!mkdir -p {aux_dir}")
            aux_download_files.add(
                f"![ -f {os.path.relpath(aux_file, dirname)} ] || "
                + f"wget {aux_url} -O {os.path.relpath(aux_file, dirname)}")
    if len(aux_create_dirs) + len(aux_download_files) > 0:
        aux_create_dirs = "\n".join(sorted(aux_create_dirs))
        aux_download_files = "\n".join(sorted(aux_download_files))
        auxiliary_files_cell = f"""# Download data files
{aux_create_dirs}
{aux_download_files}"""
        auxiliary_files_cell = nbformat.v4.new_code_cell(auxiliary_files_cell)
        auxiliary_files_cell.id = "auxiliary_files"
        nb.cells.insert(insert_base + 2, auxiliary_files_cell)

# Add the links of all other notebooks to auxiliary urls
for link_file in glob.glob(os.path.join(top_dir, "**", "*.ipynb.link"), recursive=True):
    with io.open(link_file, "r", encoding="utf8") as f:
        link = f.read()
    notebook = link_file.replace(".ipynb.link", ".ipynb")
    aux_urls[os.path.relpath(notebook, top_dir)] = link
    aux_urls[os.path.relpath(notebook, dirname)] = link

# 4. Update images and links
add_quotes_or_parentheses = (
    lambda text: '"' + text + '"',
    lambda text: "'" + text + "'",
    lambda text: "(" + text + ")"
)
for cell in nb.cells:
    if cell.cell_type == "markdown":
        for (original, url) in aux_urls.items():
            for preprocess in add_quotes_or_parentheses:
                cell.source = cell.source.replace(preprocess(original), preprocess(url))

# Overwrite notebook content
with io.open(filename, "w", encoding="utf8") as f:
    nbformat.write(nb, f)
