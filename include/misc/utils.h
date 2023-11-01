#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

class StopWatch {
public:
    StopWatch(bool active = true) : active(active), is_running(false) {}

    void start(std::string name) {
        if (active && !is_running) {
            m_name = name;
            start_time = std::chrono::high_resolution_clock::now();
            is_running = true;
        }
    }

    void stop() {
        if (active && is_running) {
            end_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end_time - start_time).count();
            std::cout << std::fixed << std::setprecision(3) << "[" << m_name << "] " << "Elapsed time: " << elapsed  << " s" << std::endl;

            // Reset the stopwatch
            is_running = false;
            start_time = std::chrono::high_resolution_clock::time_point();
            end_time = std::chrono::high_resolution_clock::time_point();
        }
    }

private:
    bool active, is_running;
    std::string m_name;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};