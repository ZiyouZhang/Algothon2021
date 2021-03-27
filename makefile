classifier: src/classifier.cpp
	@g++ $^ -o classifier

run: classifier
	@./classifier

.phony: clean
make clean:
	rm -rf classifier