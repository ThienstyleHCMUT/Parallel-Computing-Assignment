CXX      ?= g++
MPICXX   ?= mpicxx
BASE     = -O3 -Wall -Wextra -std=c++17 -Iinclude
SRC      = src/matrix.cpp src/algorithms.cpp src/bench_main.cpp

ifeq ($(OS),Windows_NT)
    EXT = .exe
    RM  = del /f /q
else
    EXT =
    RM  = rm -f
endif

all: bin/bench_seq$(EXT) bin/bench_omp$(EXT) test_matrix$(EXT)
mpi: bin/bench_mpi$(EXT) bin/bench_hybrid$(EXT)
everything: all mpi

bin:
	mkdir -p bin

bin/bench_seq$(EXT): $(SRC) | bin
	$(CXX) $(BASE) -Wno-unknown-pragmas $(SRC) -o $@

bin/bench_omp$(EXT): $(SRC) | bin
	$(CXX) $(BASE) -fopenmp $(SRC) -o $@

bin/bench_mpi$(EXT): $(SRC) | bin
	$(MPICXX) $(BASE) -DUSE_MPI -Wno-unknown-pragmas $(SRC) -o $@

bin/bench_hybrid$(EXT): $(SRC) | bin
	$(MPICXX) $(BASE) -DUSE_MPI -fopenmp $(SRC) -o $@

test_matrix$(EXT): src/matrix.cpp src/algorithms.cpp test/test_matrix.cpp
	$(CXX) $(BASE) -fopenmp $^ -o $@

test: test_matrix$(EXT)
	./test_matrix$(EXT)

clean:
	$(RM) test_matrix$(EXT) benchmark_test.csv bin/bench_*

.PHONY: all mpi everything test clean
