CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Iinclude -fopenmp

ifeq ($(OS),Windows_NT)
    TARGET = test_matrix.exe
    RM = del /f /q
else
    TARGET = test_matrix
    RM = rm -f
endif

all: $(TARGET)

$(TARGET): src/matrix.cpp test/test_matrix.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(TARGET) benchmark_test.csv
