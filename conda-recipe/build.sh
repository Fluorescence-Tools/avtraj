#!/usr/bin/env bash
# build documentation
#sphinx-apidoc -o doc avtraj
#cd doc
#make html
#cd ..

$PYTHON setup.py install --single-version-externally-managed --record=record.txt  # Python command to install the script.
#python setup.py sdist
#twine upload dist/*
