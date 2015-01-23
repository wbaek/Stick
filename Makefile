NUMBER_OF_BUILD_THREAD=16

all:build

prepare:

help:
	make -C src

build: prepare
	make --no-print-directory -j $(NUMBER_OF_BUILD_THREAD) -C src build
	make --no-print-directory -j $(NUMBER_OF_BUILD_THREAD) -C main build

rebuild: clean build

install: build
	make -C src install
	make -C main install
	
clean:
	make -C src clean
	make -C main clean
	make -C test clean
	
#for unit test
test-prepare: build
	sh ./scripts/install/install_gtest.sh
	make -j $(NUMBER_OF_BUILD_THREAD) -C test build

test: check
check: test-prepare
	make -j $(NUMBER_OF_BUILD_THREAD) -C test run

test-travis: build
	make -j $(NUMBER_OF_BUILD_THREAD) -C test run

