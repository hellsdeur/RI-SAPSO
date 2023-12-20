.PHONY: downloads compile

downloads:
	wget https://raw.githubusercontent.com/lacerdamarcelo/cec17_python/master/cec17_test_func.c
	wget https://raw.githubusercontent.com/lacerdamarcelo/cec17_python/master/cec17_functions.py

compile:
	gcc -fPIC -shared -lm -o cec17_test_func.so cec17_test_func.c

all: downloads compile
	echo All configuration steps complete.