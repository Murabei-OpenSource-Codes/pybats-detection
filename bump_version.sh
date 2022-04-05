nosetests -v ../tests/test__smooth.py
nosetests -v ../tests/test__intervention.py
nosetests -v ../tests/test__monitor.py


source VERSION
sed -e 's#{VERSION}#'"${VERSION}"'#g' setup_template.cfg > setup.cfg

rm -R dist
rm -R build
python3 setup.py build sdist bdist_wheel
# git add --all
# git commit -m "Bumped version number to ${VERSION}"
# git tag -a ${VERSION} -m "Bumped version number to ${VERSION}"
# git push
# git push origin ${VERSION}
