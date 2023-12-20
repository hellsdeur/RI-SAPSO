.PHONY: downloads compile

downloads:
	git clone git@github.com:oakboat/cec17-python.git
	mv cec17-python/input_data ./
	mv cec17-python/cec* ./
	rm -rf cec17-python

# compile:
# 	gcc -fPIC -shared -lm -o cec17_test_func.so cec17_test_func.c

clean:
	rm -rf input_data/
	rm -rf cec*

all: downloads
	echo All configuration steps complete.