#pragma once

#include <functional>
#include <vector>

float add(float a, float b);
float mul(float a, float b);

float combine3(
    std::function<float(float, float)> fn, float a, float b, float c);

std::function<float(float, float, float)> combine3(
    std::function<float(float, float)> fn);

std::function<std::vector<float>(const std::vector<float>&)> filter(
    std::function<bool(float)> fn);
