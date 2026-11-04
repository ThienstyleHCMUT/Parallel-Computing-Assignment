CXX = g++
MPICXX = mpic++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Iinclude -fopenmp

ifeq ($(OS),Windows_NT)
    TARGET = test_matrix.exe
    TARGET_MPI = test_mpi.exe
    RM = del /f /q
else
    TARGET = test_matrix
    TARGET_MPI = test_mpi
    RM = rm -f
endif

all: $(TARGET)

$(TARGET): src/matrix.cpp test/test_matrix.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# Target mới dành cho kiểm thử MPI
test_mpi: src/matrix.cpp test/test_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $^ -o $(TARGET_MPI)

clean:
	$(RM) $(TARGET) $(TARGET_MPI) benchmark_test.csv
