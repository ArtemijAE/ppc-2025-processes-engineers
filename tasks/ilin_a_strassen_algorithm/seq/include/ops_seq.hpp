#pragma once

#include "ilin_a_strassen_algorithm/common/include/common.hpp"

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

  std::vector<double> strassenMultiply(const std::vector<double> &A,
                                       const std::vector<double> &B, int n);
  
  std::vector<double> naiveMultiply(const std::vector<double> &A,
                                    const std::vector<double> &B, int n);
  
  std::vector<std::vector<double>> computeStrassenProducts(
      const std::vector<double> &A11, const std::vector<double> &A12,
      const std::vector<double> &A21, const std::vector<double> &A22,
      const std::vector<double> &B11, const std::vector<double> &B12,
      const std::vector<double> &B21, const std::vector<double> &B22,
      int half);
  
  void computeResultSubmatrices(const std::vector<std::vector<double>> &products,
                                std::vector<double> &C11, std::vector<double> &C12,
                                std::vector<double> &C21, std::vector<double> &C22,
                                int half);
  
  void addMatrix(const std::vector<double> &A, const std::vector<double> &B,
                 std::vector<double> &C, int n);
  void subtractMatrix(const std::vector<double> &A, const std::vector<double> &B,
                      std::vector<double> &C, int n);
  void splitMatrix(const std::vector<double> &A, std::vector<double> &A11,
                   std::vector<double> &A12, std::vector<double> &A21,
                   std::vector<double> &A22, int n);
  void joinMatrix(std::vector<double> &A, const std::vector<double> &A11,
                  const std::vector<double> &A12, const std::vector<double> &A21,
                  const std::vector<double> &A22, int n);

  int original_size_;
  int padded_size_;
};

}  // namespace ilin_a_strassen_algorithm