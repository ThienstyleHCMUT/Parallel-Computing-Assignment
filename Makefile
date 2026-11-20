CXX = g++
MPICXX = mpic++

CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Iinclude
OMPFLAGS = -fopenmp

COMMON_SRC = src/matrix.cpp src/algorithms.cpp src/bench_main.cpp

ifeq ($(OS),Windows_NT)
    TARGET = test_matrix.exe
    TARGET_MPI = test_mpi.exe
    RM = del /f /q
else
    TARGET = test_matrix
    TARGET_MPI = test_mpi
    RM = rm -f
endif

.PHONY: all clean

# Default target
all: $(TARGET)

# Sequential benchmark
bench_seq: $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas $^ -o $@

# OpenMP benchmark
bench_omp: $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

# MPI benchmark
bench_mpi: $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) -Wno-unknown-pragmas -DUSE_MPI $^ -o $@

# Hybrid MPI + OpenMP benchmark
bench_hybrid: $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DUSE_MPI $^ -o $@

# Matrix unit test
$(TARGET): src/matrix.cpp test/test_matrix.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# MPI sanity test
test_mpi: src/matrix.cpp test/test_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $^ -o $(TARGET_MPI)

clean:
	$(RM) $(TARGET) $(TARGET_MPI) \
		bench_seq bench_omp bench_mpi bench_hybrid \
		benchmark_test.csv
