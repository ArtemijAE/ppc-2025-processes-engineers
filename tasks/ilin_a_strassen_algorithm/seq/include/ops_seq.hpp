#pragma once

#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ilin_a_strassen_algorithm {

class IlinAStrassenAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit IlinAStrassenAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> StrassenMultiply(const std::vector<double> &a, const std::vector<double> &b, int n);
  static std::vector<double> NaiveMultiply(const std::vector<double> &a, const std::vector<double> &b, int n);
  std::vector<std::vector<double>> ComputeStrassenProducts(
      const std::vector<double> &a11, const std::vector<double> &a12, const std::vector<double> &a21,
      const std::vector<double> &a22, const std::vector<double> &b11, const std::vector<double> &b12,
      const std::vector<double> &b21, const std::vector<double> &b22, int half);
  static void ComputeResultSubmatrices(const std::vector<std::vector<double>> &products, std::vector<double> &c11,
                                       std::vector<double> &c12, std::vector<double> &c21, std::vector<double> &c22,
                                       int half);
  static void AddMatrix(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n);
  static void SubtractMatrix(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n);
  static void SplitMatrix(const std::vector<double> &a, std::vector<double> &a11, std::vector<double> &a12,
                          std::vector<double> &a21, std::vector<double> &a22, int n);
  static void JoinMatrix(std::vector<double> &a, const std::vector<double> &a11, const std::vector<double> &a12,
                         const std::vector<double> &a21, const std::vector<double> &a22, int n);

  int original_size_{};
  int padded_size_{};
  static constexpr int kThreshold = 64;
};

}  // namespace ilin_a_strassen_algorithm
