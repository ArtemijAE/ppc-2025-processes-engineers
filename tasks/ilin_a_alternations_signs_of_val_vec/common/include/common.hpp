#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ilin_a_alternations_signs_of_val_vec
