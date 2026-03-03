#!/bin/env bash

set -e 

cd $(dirname $0)/..

rm -frv notebooks
mkdir notebooks

vision_toolkit_version_identifier="$(setuptools-scm | cut -d. -f1-4)"
vision_toolkit_pyindex="https://${REPO_OWNER}.github.io/${REPO_SHORTNAME}/pyindex"

for md in notebooks_original/*.md; do
    ipynb=$(echo $(basename $md) | sed -e 's/\.md/\.ipynb/')
    preview=$(echo $(basename $md) | sed -e 's/\.md/_preview\.html/')
    echo -e "[Preview notebook](./$preview) | [Download notebook](./$ipynb)\n" |
	cat - $md |
	awk "{gsub(\"__VISION_TOOLKIT_PYINDEX__\", \"${vision_toolkit_pyindex}\"); print}" |
	awk "{gsub(\"__VISION_TOOLKIT_VERSION_IDENTIFIER__\", \"${vision_toolkit_version_identifier}\"); print}" \
	    > notebooks/$(basename $md)
done

jupytext notebooks/*.md --to ipynb
PYTHONSTARTUP="$PWD/notebooks_original/notebook_startup.py" jupyter-execute --inplace notebooks/*.ipynb
jupyter-nbconvert --to html notebooks/*.ipynb --output '{notebook_name}_preview'
jupyter-nbconvert --to markdown notebooks/*.ipynb --template=notebooks_original/nbconvert_zensical.tpl
	  
