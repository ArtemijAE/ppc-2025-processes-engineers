#pragma once

#include <vector>

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

  static int CountLocalSignChanges(const std::vector<int> &segment);
  static BoundaryInfo GatherEdgeValues(const std::vector<int> &segment);
  static int CountEdgeAlternations(const BoundaryInfo &edges, int total_processes);
  static void CalculateDistribution(int data_size, int world_size, std::vector<int> &counts, std::vector<int> &offsets);
  static void DistributeData(const std::vector<int> &global_data, std::vector<int> &local_data, int world_rank,
                             int world_size);
};

}  // namespace ilin_a_alternations_signs_of_val_vec
