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

  std::vector<double> multiplyMatrices(const std::vector<double> &A, const std::vector<double> &B, int n);
  std::vector<double> distributedNaiveMultiply(const std::vector<double> &A, const std::vector<double> &B, int n);
  std::vector<double> parallelStrassen(const std::vector<double> &A, const std::vector<double> &B, int n);
  std::vector<double> parallelStrassenRecursive(const std::vector<double> &A, const std::vector<double> &B, int n);
  std::vector<double> strassenSequential(const std::vector<double> &A, const std::vector<double> &B, int n);
  std::vector<double> naiveMultiplySeq(const std::vector<double> &A, const std::vector<double> &B, int n);
  void addMatrix(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, int n);
  void subtractMatrix(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, int n);
  void splitMatrix(const std::vector<double> &A, std::vector<double> &A11, std::vector<double> &A12,
                   std::vector<double> &A21, std::vector<double> &A22, int n);
  void joinMatrix(std::vector<double> &A, const std::vector<double> &A11, const std::vector<double> &A12,
                  const std::vector<double> &A21, const std::vector<double> &A22, int n);
  std::tuple<int, int> calculateMatrixRange(int total_matrices) const;
  void computeSingleProduct(int product_idx, const std::vector<double> &A11, const std::vector<double> &A12,
                            const std::vector<double> &A21, const std::vector<double> &A22,
                            const std::vector<double> &B11, const std::vector<double> &B12,
                            const std::vector<double> &B21, const std::vector<double> &B22, int half,
                            std::vector<double> &result);
  void gatherProductMatrix(const std::vector<double> &local_product, std::vector<double> &buffer);
  void mergeProductFromBuffer(const std::vector<double> &buffer, int proc_count, int matrix_size,
                              std::vector<double> &product);
  void computeResultFromProducts(const std::vector<double> &P1, const std::vector<double> &P2,
                                 const std::vector<double> &P3, const std::vector<double> &P4,
                                 const std::vector<double> &P5, const std::vector<double> &P6,
                                 const std::vector<double> &P7, int half, std::vector<double> &C11,
                                 std::vector<double> &C12, std::vector<double> &C21, std::vector<double> &C22);
  std::tuple<int, int, int> calculateRowDistribution(int n) const;
  std::vector<double> computeLocalRows(const std::vector<double> &A, const std::vector<double> &B, int n, int start_row,
                                       int local_rows);
  void setupGatherParameters(int n, std::vector<int> &recvcounts, std::vector<int> &displs) const;
  std::vector<double> gatherLocalResults(const std::vector<double> &local_result, int n,
                                         const std::vector<int> &recvcounts, const std::vector<int> &displs);
  std::vector<double> reorderGatheredResults(const std::vector<double> &gathered_data, int n,
                                             const std::vector<int> &recvcounts, const std::vector<int> &displs);
  void prepareSmallMatricesCase(std::vector<double> &A_full, std::vector<double> &B_full,
                                std::vector<double> &final_result);
  void prepareLargeMatricesCase(std::vector<double> &A_full, std::vector<double> &B_full,
                                std::vector<double> &final_result);
  void distributeFinalResult(std::vector<double> &final_result);

  int world_size_;
  int world_rank_;
  int original_size_;
  int padded_size_;
  int threshold_;
};

}  // namespace ilin_a_strassen_algorithm
