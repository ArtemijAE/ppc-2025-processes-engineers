#pragma once

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

  MatrixData data_;
  std::vector<double> solution_;

  int rank_;
  int size_;
  int n_;
  int band_;
  int local_rows_;
  int row_start_;
  int row_end_;
  int rows_per_proc_;
  int remainder_;
  std::vector<double> local_matrix_;
  std::vector<double> local_vector_;
};

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
