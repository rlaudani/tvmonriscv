#include <fstream>
#include "stdio.h"
#include "time.h"
#include <iostream>
#include <vector>

struct timespec start, stop;
std::vector<timespec> start_times;
std::vector<timespec> stop_times;

extern "C"
void my_test_start(int id) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("START: %d", id);
}

extern "C"
void my_test_end(int id) {
    clock_gettime(CLOCK_MONOTONIC, &stop);
    start_times.push_back(start);
    stop_times.push_back(stop);
    printf("STOP: %d", id);
}

__attribute__((constructor))
void go() {
    // TODO: Find better option for reservation
    start_times.reserve(5);
    stop_times.reserve(5);
    return;
}

__attribute__((destructor))
void save_timestamps() {
    FILE *file = fopen("compile/timestamps.txt", "a");
    if (file == NULL) {
        printf("Error opening file!");
        return;
    }
    for (size_t i = 0; i < start_times.size(); ++i) {
        long long elapsed_ns = (stop_times[i].tv_sec - start_times[i].tv_sec) * 1000000000LL 
                             + (stop_times[i].tv_nsec - start_times[i].tv_nsec);
        fprintf(file, "%d, %lld\n", i, elapsed_ns);
    }
    fclose(file);
    return;
}
