cd ./python
CMAKE_BUILD_PARALLEL_LEVEL=$(cat /proc/cpuinfo | grep processor | wc -l) python3 setup.py develop
cd -