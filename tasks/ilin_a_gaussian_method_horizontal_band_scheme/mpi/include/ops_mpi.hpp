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

  void ProcessForwardElimination();
  void ProcessBackwardSubstitution();

  double FindLocalPivotValue(int k, int &local_max_row) const;
  void BroadcastPivotData(int pivot_owner, std::vector<double> &pivot_row, double &pivot_b, int global_max_row) const;
  [[nodiscard]] int CalculateRowOwner(int row) const;
  [[nodiscard]] int CalculatePivotOwner(int global_max_row) const;
  [[nodiscard]] int FindLocalRowIndex(int global_row) const;

  void HandleRowSwapLocal(int k_local_idx, int pivot_owner, int global_max_row);
  void HandleRowSwapRemote(int k_owner, int global_max_row);
  void SwapRowsLocally(int k_local_idx, int pivot_local_idx);
  void ExchangeRowsWithRemote(int k_local_idx, int pivot_owner);
  void ReceiveRowFromRemote(int pivot_local_idx, int k_owner);

  void EliminateRow(int i, int k, const std::vector<double> &pivot_row, double pivot_b);
  void UpdateRowValues(int i, int k, double factor, const std::vector<double> &pivot_row);

  void GatherAllData(std::vector<double> &recv_matrix, std::vector<double> &recv_vector);
  void ReconstructFullMatrix(const std::vector<double> &recv_matrix, const std::vector<double> &recv_vector,
                             std::vector<double> &full_matrix, std::vector<double> &full_vector) const;
  void SolveBackwardSubstitution(const std::vector<double> &full_matrix, const std::vector<double> &full_vector);

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
