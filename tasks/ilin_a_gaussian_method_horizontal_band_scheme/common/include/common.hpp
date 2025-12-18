#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

struct MatrixData {
  std::vector<double> matrix;
  std::vector<double> vector;
  int size = 0;
  int band_width = 0;
};

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
