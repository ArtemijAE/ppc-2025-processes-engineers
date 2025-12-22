#pragma once

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ilin_a_strassen_algorithm {

class IlinAStrassenAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit IlinAStrassenAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  std::vector<double> multiplyMatrices(const std::vector<double> &A,
                                       const std::vector<double> &B, int n);
  
  std::vector<double> distributedNaiveMultiply(const std::vector<double> &A,
                                               const std::vector<double> &B, int n);
  
  std::vector<double> parallelStrassen(const std::vector<double> &A,
                                       const std::vector<double> &B, int n);
  
  std::vector<double> parallelStrassenRecursive(const std::vector<double> &A,
                                                const std::vector<double> &B, int n);
  
  std::vector<double> strassenSequential(const std::vector<double> &A,
                                         const std::vector<double> &B, int n);
  
  std::vector<double> naiveMultiplySeq(const std::vector<double> &A,
                                       const std::vector<double> &B, int n);
  
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

  int world_size_;
  int world_rank_;
  int original_size_;
  int padded_size_;
  int threshold_;
};

}  // namespace ilin_a_strassen_algorithm