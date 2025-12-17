#include "ilin_a_alternations_signs_of_val_vec/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

IlinAAlternationsSignsOfValVecSEQ::IlinAAlternationsSignsOfValVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IlinAAlternationsSignsOfValVecSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool IlinAAlternationsSignsOfValVecSEQ::PreProcessingImpl() {
  return true;
}

bool IlinAAlternationsSignsOfValVecSEQ::RunImpl() {
  const std::vector<int> &vec = GetInput();
  int alternation_count = 0;

  if (vec.size() < 2) {
    GetOutput() = 0;
    return true;
  }
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    if ((vec[i] < 0 && vec[i + 1] >= 0) || (vec[i] >= 0 && vec[i + 1] < 0)) {
      alternation_count++;
    }
  }
  GetOutput() = alternation_count;
  return true;
}

bool IlinAAlternationsSignsOfValVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_alternations_signs_of_val_vec
