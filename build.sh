$PYTHON setup.py install --single-version-externally-managed --record=record.txt  # Python command to install the script.
$PYTHON setup.py sdist
twine upload dist/*
