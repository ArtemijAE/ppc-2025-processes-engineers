#pragma once

#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

class IlinAGaussianMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IlinAGaussianMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ForwardElimination(int n, int m, std::vector<double> &matrix, std::vector<double> &b);
  [[nodiscard]] static int FindPivotRow(int k, int n, int m, const std::vector<double> &matrix);
  static void SwapRows(int row1, int row2, int m, std::vector<double> &matrix, std::vector<double> &b);
  static void EliminateRow(int i, int k, int m, std::vector<double> &matrix, std::vector<double> &b, double pivot);
  void BackwardSubstitution(int n, int m, const std::vector<double> &matrix, const std::vector<double> &b);

  MatrixData data_;
  std::vector<double> solution_;
};

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
