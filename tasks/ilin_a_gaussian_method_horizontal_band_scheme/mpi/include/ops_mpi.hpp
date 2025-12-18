#pragma once

#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

class IlinAGaussianMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IlinAGaussianMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void InitializeMPI();
  void BroadcastInputData();
  void ScatterLocalData();
  void FindGlobalPivot(int k, int &global_max_row, double &global_max_val, std::vector<double> &pivot_row,
                       double &pivot_b, int &pivot_owner) const;
  void SwapRowsIfNeeded(int k, int global_max_row, int pivot_owner);
  void EliminateRows(int k, const std::vector<double> &pivot_row, double pivot_b);
  void GatherResults();
  void BackSubstitution();

  MatrixData data_;
  std::vector<double> solution_;

  int rank_ = 0;
  int size_ = 1;
  int n_ = 0;
  int band_ = 0;
  int local_rows_ = 0;
  int row_start_ = 0;
  int row_end_ = 0;
  int rows_per_proc_ = 0;
  int remainder_ = 0;
  std::vector<double> local_matrix_;
  std::vector<double> local_vector_;
  std::vector<double> pivot_row_buf_;
};

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
