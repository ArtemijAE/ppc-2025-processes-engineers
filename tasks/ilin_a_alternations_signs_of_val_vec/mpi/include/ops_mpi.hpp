#pragma once

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

struct BoundaryInfo {
  std::vector<int> all_edges;
};

class IlinAAlternationsSignsOfValVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IlinAAlternationsSignsOfValVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int CountLocalSignChanges(const std::vector<int> &segment);
  BoundaryInfo GatherEdgeValues(const std::vector<int> &segment);
  int CountEdgeAlternations(const BoundaryInfo &edges, int total_processes);
};

}  // namespace ilin_a_alternations_signs_of_val_vec
