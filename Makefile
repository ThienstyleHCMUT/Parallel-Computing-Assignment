CXX      ?= g++
MPICXX   ?= mpicxx
BASE     = -O3 -Wall -Wextra -std=c++17 -Iinclude
SRC      = src/matrix.cpp src/algorithms.cpp src/bench_main.cpp
TEST_SRC = src/matrix.cpp src/algorithms.cpp src/test_strassen.cpp

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

# root-level copies used by scripts/benchmark.slurm (make bench_seq bench_omp bench_mpi bench_hybrid)
bench_seq bench_omp bench_mpi bench_hybrid: %: bin/%$(EXT)
	cp $< $@

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

bin/test_strassen_seq$(EXT): $(TEST_SRC) | bin
	$(CXX) $(BASE) -Wno-unknown-pragmas $(TEST_SRC) -o $@

bin/test_strassen_omp$(EXT): $(TEST_SRC) | bin
	$(CXX) $(BASE) -fopenmp $(TEST_SRC) -o $@

test_matrix$(EXT): src/matrix.cpp src/algorithms.cpp test/test_matrix.cpp
	$(CXX) $(BASE) -fopenmp $^ -o $@

tests: bin/test_strassen_seq$(EXT) bin/test_strassen_omp$(EXT)

test: tests
	./bin/test_strassen_seq$(EXT)
	./bin/test_strassen_omp$(EXT)

clean:
	$(RM) test_matrix$(EXT) benchmark_test.csv bin/bench_* bin/test_strassen_* bench_seq bench_omp bench_mpi bench_hybrid

.PHONY: all mpi everything tests test clean
