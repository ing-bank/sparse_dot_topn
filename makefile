.PHONY: install
install:
		echo "Install Cython and wbaa-utils"
		rm -rf build
		rm -rf dist
		rm -rf wbaa_utils.egg-info
		pip install -r requirements.txt
		python setup.py build_ext -i -v
		python setup.py install

uninstall:
		python setup.py build_ext -i -v
		python setup.py install
		python setup.py install --record files.txt
		cat files.txt | xargs rm -rf
		rm files.txt
		rm -rf build
		rm -rf dist
		rm -rf wbaa_utils.egg-info
