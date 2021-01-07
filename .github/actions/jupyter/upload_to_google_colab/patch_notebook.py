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
except ImportError as e:
    !apt-get install -y -qq software-properties-common
    !add-apt-repository -y ppa:fenics-packages/fenics
    !apt-get update -qq
    !apt install -y --no-install-recommends fenics
    !sed -i "s|#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 8 && PETSC_VERSION_RELEASE == 1|#if 1|" /usr/include/dolfin/la/PETScLUSolver.h
    !pip3 -q install --upgrade sympy
    exec(open("/usr/lib/python3/dist-packages/dolfin.pth").read())
    import dolfin"""  # noqa: E501
nb.cells.insert(insert_base, nbformat.v4.new_code_cell(fenics_installation_cell))

# 2. Add RBniCS installation cell
rbnics_installation_cell = """# Install RBniCS
try:
    import rbnics
except ImportError as e:
    !pip3 -q install --upgrade cvxopt multipledispatch pylru toposort
    ![ -d "/tmp/RBniCS" ] || git clone https://github.com/RBniCS/RBniCS /tmp/RBniCS
    !cd /tmp/RBniCS && python3 setup.py install && cd -
    !ln -s /usr/local/lib/python3.6/dist-packages/RBniCS*egg/rbnics /usr/local/lib/python3.6/dist-packages/
    import rbnics
import rbnics.utils.config
assert "dolfin" in rbnics.utils.config.config.get("backends", "required backends")"""
nb.cells.insert(insert_base + 1, nbformat.v4.new_code_cell(rbnics_installation_cell))

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
            aux_url = f"https://github.com/RBniCS/RBniCS/raw/{os.getenv('BRANCH', 'master')}/{aux_real_file}"
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
        aux_create_dirs = "\n".join(aux_create_dirs)
        aux_download_files = "\n".join(aux_download_files)
        auxiliary_files_cell = f"""# Download data files
{aux_create_dirs}
{aux_download_files}"""
        nb.cells.insert(insert_base + 2, nbformat.v4.new_code_cell(auxiliary_files_cell))

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
