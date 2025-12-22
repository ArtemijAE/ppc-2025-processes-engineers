#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

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

  std::vector<double> MultiplyMatrices(const std::vector<double> &a, const std::vector<double> &b, int n);
  std::vector<double> DistributedNaiveMultiply(const std::vector<double> &a, const std::vector<double> &b, int n);
  std::vector<double> ParallelStrassen(const std::vector<double> &a, const std::vector<double> &b, int n);
  std::vector<double> ParallelStrassenRecursive(const std::vector<double> &a, const std::vector<double> &b, int n);
  std::vector<double> StrassenSequential(const std::vector<double> &a, const std::vector<double> &b, int n);
  static std::vector<double> NaiveMultiplySeq(const std::vector<double> &a, const std::vector<double> &b, int n);
  static void AddMatrix(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n);
  static void SubtractMatrix(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n);
  static void SplitMatrix(const std::vector<double> &a, std::vector<double> &a11, std::vector<double> &a12,
                          std::vector<double> &a21, std::vector<double> &a22, int n);
  static void JoinMatrix(std::vector<double> &a, const std::vector<double> &a11, const std::vector<double> &a12,
                         const std::vector<double> &a21, const std::vector<double> &a22, int n);
  [[nodiscard]] std::tuple<int, int> CalculateMatrixRange(int total_matrices) const;
  void ComputeSingleProduct(int product_idx, const std::vector<double> &a11, const std::vector<double> &a12,
                            const std::vector<double> &a21, const std::vector<double> &a22,
                            const std::vector<double> &b11, const std::vector<double> &b12,
                            const std::vector<double> &b21, const std::vector<double> &b22, int half,
                            std::vector<double> &result);
  void GatherProductMatrix(const std::vector<double> &local_product, std::vector<double> &buffer) const;
  static void MergeProductFromBuffer(const std::vector<double> &buffer, int proc_count, int matrix_size,
                                     std::vector<double> &product);
  static void ComputeResultFromProducts(const std::vector<double> &p1, const std::vector<double> &p2,
                                        const std::vector<double> &p3, const std::vector<double> &p4,
                                        const std::vector<double> &p5, const std::vector<double> &p6,
                                        const std::vector<double> &p7, int half, std::vector<double> &c11,
                                        std::vector<double> &c12, std::vector<double> &c21, std::vector<double> &c22);
  [[nodiscard]] std::tuple<int, int, int> CalculateRowDistribution(int n) const;
  static std::vector<double> ComputeLocalRows(const std::vector<double> &a, const std::vector<double> &b, int n,
                                              int start_row, int local_rows);
  void SetupGatherParameters(int n, std::vector<int> &recvcounts, std::vector<int> &displs) const;
  std::vector<double> GatherLocalResults(const std::vector<double> &local_result, int n,
                                         const std::vector<int> &recvcounts, const std::vector<int> &displs) const;
  std::vector<double> ReorderGatheredResults(const std::vector<double> &gathered_data, int n) const;
  void PrepareSmallMatricesCase(std::vector<double> &a_full, std::vector<double> &b_full,
                                std::vector<double> &final_result);
  void PrepareLargeMatricesCase(std::vector<double> &a_full, std::vector<double> &b_full,
                                std::vector<double> &final_result);
  void DistributeFinalResult(std::vector<double> &final_result);

  int world_size_{};
  int world_rank_{};
  int original_size_{};
  int padded_size_{};
  static constexpr int kThreshold = 64;
};

}  // namespace ilin_a_strassen_algorithm
