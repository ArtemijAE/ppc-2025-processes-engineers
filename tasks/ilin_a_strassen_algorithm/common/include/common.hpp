#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ilin_a_strassen_algorithm {

struct MatrixData {
  std::vector<double> A;
  std::vector<double> B;
  int size;
};

struct ResultData {
  std::vector<double> C;
  int size;
};

using InType = MatrixData;
using OutType = ResultData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ilin_a_strassen_algorithm
