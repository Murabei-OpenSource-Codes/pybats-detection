cd src
nosetests -v ../tests/test__smooth.py
nosetests -v ../tests/test__intervention.py
nosetests -v ../tests/test__monitor.py
cd ..

source VERSION
sed -e 's#{VERSION}#'"${VERSION}"'#g' setup_template.cfg > setup.cfg

rm -R dist
rm -R build
python3 setup.py build sdist bdist_wheel

# twine upload --repository testpypi dist/*
# twine upload dist/*

# git tag -a v${VERSION} -m "Bumped version number to ${VERSION}"
# git push origin v${VERSION}
