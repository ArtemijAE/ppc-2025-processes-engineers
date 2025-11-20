#pragma once

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

class IlinAAlternationsSignsOfValVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IlinAAlternationsSignsOfValVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ilin_a_alternations_signs_of_val_vec
