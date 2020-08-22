#!/bin/bash
set -e
src=${1?"Please provide input notebook as first argument"}
dest=${FOLDER_PATH}-colab/$(dirname ${src#${FOLDER_PATH}/})/$(basename $src)
mkdir -p $(dirname $dest)
mv $src $dest

# Use absolute path for images
sed -i "s|<img src=\\\\\"data|<img src=\\\\\"https://github.com/RBniCS/RBniCS/raw/${BRANCH}/$(dirname $src)/data|g" $dest

cat <<EOF > 01-setup-fenics.py
# Install FEniCS
try:
    import dolfin
except ImportError as e:
    !apt-get install -y -qq software-properties-common
    !add-apt-repository -y ppa:fenics-packages/fenics
    !apt-get update -qq
    !apt install -y --no-install-recommends fenics
    !sed -i "s|#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 8 && PETSC_VERSION_RELEASE == 1|#if 1|" /usr/include/dolfin/la/PETScLUSolver.h
    !pip3 -q install --upgrade sympy
    import dolfin
EOF

cat <<EOF > 02-setup-rbnics.py
# Install RBniCS
try:
    import rbnics
except ImportError as e:
    !pip3 -q install --upgrade cvxopt multipledispatch pylru toposort
    ![ -d "/tmp/RBniCS" ] || git clone https://github.com/RBniCS/RBniCS /tmp/RBniCS
    !cd /tmp/RBniCS && python3 setup.py install && cd -
    !ln -s /usr/local/lib/python3.6/dist-packages/RBniCS*egg/rbnics /usr/local/lib/python3.6/dist-packages/
    import rbnics
import rbnics.utils.config
assert "dolfin" in rbnics.utils.config.config.get("backends", "required backends")
EOF

# Create auxiliary directories and download auxiliary files (e.g. meshes)
rm -rf 03-create-aux-dirs.py 04-download-aux-files.py
for aux_file in $(find $(dirname $src) \( -type f -o -type l \) -not -path "*.ipynb" -not -name .gitignore); do
    if [ -L $aux_file ]; then
        aux_link_dest=$aux_file
        while [ -L $aux_link_dest ]; do
            aux_link_dest=$(readlink -f $aux_link_dest)
        done
        if [ -f $aux_link_dest ]; then
            aux_url="https://github.com/RBniCS/RBniCS/raw/${BRANCH}/${aux_link_dest#$PWD/}"
        else
            continue
        fi
    else
        aux_url="https://github.com/RBniCS/RBniCS/raw/${BRANCH}/${aux_file}"
    fi
    aux_file="${aux_file#$(dirname $src)/}"
    aux_dir="${aux_file%/*}"
    echo "!mkdir -p ${aux_dir}" >> 03-create-aux-dirs.py
    echo "![ -f ${aux_file} ] || wget ${aux_url} -O ${aux_file}" >> 04-download-aux-files.py
done
echo "# Download data files" > 03-04-create-aux-dirs-and-download-aux-files.py
sort -u 03-create-aux-dirs.py >> 03-04-create-aux-dirs-and-download-aux-files.py
sort -u 04-download-aux-files.py >> 03-04-create-aux-dirs-and-download-aux-files.py
rm -rf 03-create-aux-dirs.py 04-download-aux-files.py

new_cells=""
for cell_file in 01-setup-fenics.py 02-setup-rbnics.py 03-04-create-aux-dirs-and-download-aux-files.py; do
    new_cells="${new_cells}
  {\n
   \"cell_type\": \"code\",\n
   \"execution_count\": null,\n
   \"metadata\": {},\n
   \"outputs\": [],\n
   \"source\": [\n"
    cell_content=$(sed -e 's/\"/\\\\"/g' -e 's/^/    \"/' -e 's/$/\\\\n\",/' -e '$s/\\\\n\",$/\"\\n/' $cell_file);
    new_cells="${new_cells}${cell_content}";
    new_cells="${new_cells}
   ]\n
  },\n"
done
perl -0777 -i -pe "s%  \},\n  \{\n   \"cell_type\": \"code\",%  \},\n$new_cells  \{\n   \"cell_type\": \"code\",%" $dest
