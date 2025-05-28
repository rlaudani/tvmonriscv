#include <fstream>
#include "stdio.h"
#include "time.h"
#include <iostream>
#include <vector>

#ifndef RUNTIME_STATS_PATH
    #define RUNTIME_STATS_PATH "compile/runtime_stats.txt"
#endif

struct timespec start_time, stop_time;
std::vector<timespec> start_times;
std::vector<timespec> stop_times;


extern "C"
void my_test_start(int id) {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
}

extern "C"
void my_test_end(int id) {
    clock_gettime(CLOCK_MONOTONIC, &stop_time);
    start_times.push_back(start_time);
    stop_times.push_back(stop_time);
}

__attribute__((constructor))
void go() {
    // TODO: Find better option for reservation
    start_times.reserve(16);
    stop_times.reserve(16);
    return;
}

__attribute__((destructor))
void save_timestamps() {
    FILE *file = fopen(RUNTIME_STATS_PATH, "a");
    if (file == NULL) {
        printf("Error opening file!");
        return;
    }
        for (size_t i = 0; i < start_times.size(); ++i) {
            int64_t elapsed_ns = (int64_t)(stop_times[i].tv_sec - start_times[i].tv_sec) * 1000000000LL
                                + (stop_times[i].tv_nsec - start_times[i].tv_nsec);
        fprintf(file, "%d, %lld\n", i, elapsed_ns);
    }
    fclose(file);
    return;
}
