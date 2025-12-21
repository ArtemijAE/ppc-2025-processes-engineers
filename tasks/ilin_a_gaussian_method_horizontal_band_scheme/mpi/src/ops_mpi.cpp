#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ilin_a_gaussian_method_horizontal_band_scheme/common/include/common.hpp"

namespace ilin_a_gaussian_method_horizontal_band_scheme {

IlinAGaussianMethodMPI::IlinAGaussianMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool IlinAGaussianMethodMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  if (rank_ != 0) {
    return true;
  }

  const auto &input = GetInput();
  if (input.size() < 4) {
    return false;
  }

  int size = static_cast<int>(input[0]);
  int band_width = static_cast<int>(input[1]);

  if (size <= 0 || band_width <= 0 || band_width > size) {
    return false;
  }

  size_t expected_count = 2 + (size * band_width) + size;
  return input.size() == expected_count;
}

bool IlinAGaussianMethodMPI::PreProcessingImpl() {
  InitializeMPI();
  BroadcastInputData();
  ScatterLocalData();
  return true;
}

void IlinAGaussianMethodMPI::InitializeMPI() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

void IlinAGaussianMethodMPI::BroadcastInputData() {
  if (rank_ == 0) {
    const auto &input = GetInput();
    data_.size = static_cast<int>(input[0]);
    data_.band_width = static_cast<int>(input[1]);
  }

  MPI_Bcast(&data_.size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data_.band_width, 1, MPI_INT, 0, MPI_COMM_WORLD);

  n_ = data_.size;
  band_ = data_.band_width;

  rows_per_proc_ = n_ / size_;
  remainder_ = n_ % size_;

  if (rank_ < remainder_) {
    local_rows_ = rows_per_proc_ + 1;
    row_start_ = rank_ * local_rows_;
  } else {
    local_rows_ = rows_per_proc_;
    row_start_ = (remainder_ * (rows_per_proc_ + 1)) + ((rank_ - remainder_) * rows_per_proc_);
  }
  row_end_ = row_start_ + local_rows_;

  if (rank_ == 0) {
    const auto &input = GetInput();
    int mat_size = n_ * band_;
    data_.matrix.resize(static_cast<size_t>(mat_size));
    data_.vector.resize(static_cast<size_t>(n_));

    std::copy(input.begin() + 2, input.begin() + 2 + mat_size, data_.matrix.begin());
    std::copy(input.begin() + 2 + mat_size, input.end(), data_.vector.begin());
  } else {
    data_.matrix.resize(static_cast<size_t>(n_) * static_cast<size_t>(band_));
    data_.vector.resize(static_cast<size_t>(n_));
  }

  MPI_Bcast(data_.matrix.data(), n_ * band_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(data_.vector.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void IlinAGaussianMethodMPI::ScatterLocalData() {
  local_matrix_.resize(static_cast<size_t>(local_rows_) * static_cast<size_t>(band_));
  local_vector_.resize(static_cast<size_t>(local_rows_));

  for (int i = 0; i < local_rows_; ++i) {
    int global_row = row_start_ + i;
    const auto global_row_large = static_cast<std::ptrdiff_t>(global_row);
    std::copy(data_.matrix.begin() + (global_row_large * band_),
              data_.matrix.begin() + ((global_row_large + 1) * band_),
              local_matrix_.begin() + static_cast<std::ptrdiff_t>(i) * band_);
    local_vector_[static_cast<size_t>(i)] = data_.vector[static_cast<size_t>(global_row)];
  }

  solution_.resize(static_cast<size_t>(n_), 0.0);
  pivot_row_buf_.resize(static_cast<size_t>(band_));
}

bool IlinAGaussianMethodMPI::RunImpl() {
  ProcessForwardElimination();
  ProcessBackwardSubstitution();
  GetOutput() = solution_;
  return true;
}

void IlinAGaussianMethodMPI::ProcessForwardElimination() {
  const double eps = 1e-12;
  pivot_row_buf_.assign(band_, 0.0);
  double pivot_b = 0.0;

  for (int k = 0; k < n_; ++k) {
    double local_max = 0.0;
    int local_max_row = -1;

    local_max = FindLocalPivotValue(k, local_max_row);

    struct {
      double max_val;
      int max_row;
    } local_data{.max_val = local_max, .max_row = local_max_row}, global_data{.max_val = 0.0, .max_row = -1};

    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    int global_max_row = global_data.max_row;
    double global_max_val = global_data.max_val;

    if (global_max_val < eps) {
      solution_[static_cast<size_t>(k)] = 0.0;
      continue;
    }

    int pivot_owner = CalculatePivotOwner(global_max_row);
    BroadcastPivotData(pivot_owner, pivot_row_buf_, pivot_b, global_max_row);

    if (global_max_row != k) {
      int k_owner = CalculateRowOwner(k);

      int k_local_idx = FindLocalRowIndex(k);
      if (k_owner == rank_ && k_local_idx >= 0) {
        HandleRowSwapLocal(k_local_idx, pivot_owner, global_max_row);
      } else if (pivot_owner == rank_) {
        HandleRowSwapRemote(k_owner, global_max_row);
      }
    }

    double pivot = pivot_row_buf_[static_cast<size_t>(band_ - 1)];
    if (std::fabs(pivot) < eps) {
      continue;
    }

    for (int i = 0; i < local_rows_; ++i) {
      EliminateRow(i, k, pivot_row_buf_, pivot_b);
    }
  }
}

double IlinAGaussianMethodMPI::FindLocalPivotValue(int k, int &local_max_row) const {
  double local_max = 0.0;
  local_max_row = -1;

  for (int i = 0; i < local_rows_; ++i) {
    int global_row = row_start_ + i;
    if (global_row < k) {
      continue;
    }

    int diag_idx = band_ - 1 - (global_row - k);
    if (diag_idx < 0 || diag_idx >= band_) {
      continue;
    }

    double val = std::fabs(local_matrix_[(static_cast<size_t>(i) * static_cast<size_t>(band_)) + diag_idx]);
    if (val > local_max) {
      local_max = val;
      local_max_row = global_row;
    }
  }

  return local_max;
}

void IlinAGaussianMethodMPI::BroadcastPivotData(int pivot_owner, std::vector<double> &pivot_row, double &pivot_b,
                                                int global_max_row) const {
  if (pivot_owner == rank_) {
    int local_pivot_idx = FindLocalRowIndex(global_max_row);
    if (local_pivot_idx >= 0) {
      const double *row_start = &local_matrix_[(static_cast<size_t>(local_pivot_idx) * static_cast<size_t>(band_))];
      std::ranges::copy(row_start, row_start + band_, pivot_row.begin());
      pivot_b = local_vector_[static_cast<size_t>(local_pivot_idx)];
    }
  }

  MPI_Bcast(pivot_row.data(), band_, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
  MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
}

int IlinAGaussianMethodMPI::CalculateRowOwner(int row) const {
  if (row < (remainder_ * (rows_per_proc_ + 1))) {
    return row / (rows_per_proc_ + 1);
  }
  return remainder_ + ((row - remainder_ * (rows_per_proc_ + 1)) / rows_per_proc_);
}

int IlinAGaussianMethodMPI::CalculatePivotOwner(int global_max_row) const {
  if (global_max_row < 0) {
    return -1;
  }
  return CalculateRowOwner(global_max_row);
}

int IlinAGaussianMethodMPI::FindLocalRowIndex(int global_row) const {
  for (int i = 0; i < local_rows_; ++i) {
    if (row_start_ + i == global_row) {
      return i;
    }
  }
  return -1;
}

void IlinAGaussianMethodMPI::HandleRowSwapLocal(int k_local_idx, int pivot_owner, int global_max_row) {
  if (k_local_idx < 0) {
    return;
  }

  if (pivot_owner == rank_) {
    int pivot_local_idx = FindLocalRowIndex(global_max_row);
    if (pivot_local_idx >= 0) {
      SwapRowsLocally(k_local_idx, pivot_local_idx);
    }
  } else {
    ExchangeRowsWithRemote(k_local_idx, pivot_owner);
  }
}

void IlinAGaussianMethodMPI::HandleRowSwapRemote(int k_owner, int global_max_row) {
  int pivot_local_idx = FindLocalRowIndex(global_max_row);
  if (pivot_local_idx < 0) {
    return;
  }

  ReceiveRowFromRemote(pivot_local_idx, k_owner);
}

void IlinAGaussianMethodMPI::SwapRowsLocally(int k_local_idx, int pivot_local_idx) {
  std::swap_ranges(&local_matrix_[(static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_))],
                   &local_matrix_[(static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_))] + band_,
                   &local_matrix_[(static_cast<size_t>(pivot_local_idx) * static_cast<size_t>(band_))]);
  std::swap(local_vector_[static_cast<size_t>(k_local_idx)], local_vector_[static_cast<size_t>(pivot_local_idx)]);
}

void IlinAGaussianMethodMPI::ExchangeRowsWithRemote(int k_local_idx, int pivot_owner) {
  std::vector<double> temp_row(band_);
  double temp_b = 0.0;
  MPI_Status status;

  MPI_Recv(temp_row.data(), band_, MPI_DOUBLE, pivot_owner, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(&temp_b, 1, MPI_DOUBLE, pivot_owner, 1, MPI_COMM_WORLD, &status);

  std::ranges::copy(temp_row, &local_matrix_[(static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_))]);
  local_vector_[static_cast<size_t>(k_local_idx)] = temp_b;

  MPI_Send(&local_matrix_[(static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_))], band_, MPI_DOUBLE,
           pivot_owner, 2, MPI_COMM_WORLD);
  MPI_Send(&local_vector_[static_cast<size_t>(k_local_idx)], 1, MPI_DOUBLE, pivot_owner, 3, MPI_COMM_WORLD);
}

void IlinAGaussianMethodMPI::ReceiveRowFromRemote(int pivot_local_idx, int k_owner) {
  MPI_Send(&local_matrix_[(static_cast<size_t>(pivot_local_idx) * static_cast<size_t>(band_))], band_, MPI_DOUBLE,
           k_owner, 0, MPI_COMM_WORLD);
  MPI_Send(&local_vector_[static_cast<size_t>(pivot_local_idx)], 1, MPI_DOUBLE, k_owner, 1, MPI_COMM_WORLD);

  std::vector<double> temp_row(band_);
  double temp_b = 0.0;
  MPI_Status status;

  MPI_Recv(temp_row.data(), band_, MPI_DOUBLE, k_owner, 2, MPI_COMM_WORLD, &status);
  MPI_Recv(&temp_b, 1, MPI_DOUBLE, k_owner, 3, MPI_COMM_WORLD, &status);

  std::ranges::copy(temp_row, &local_matrix_[(static_cast<size_t>(pivot_local_idx) * static_cast<size_t>(band_))]);
  local_vector_[static_cast<size_t>(pivot_local_idx)] = temp_b;
}

void IlinAGaussianMethodMPI::EliminateRow(int i, int k, const std::vector<double> &pivot_row, double pivot_b) {
  const double eps = 1e-12;
  int global_row = row_start_ + i;

  if (global_row <= k) {
    return;
  }

  int factor_idx = band_ - 1 - (global_row - k);
  if (factor_idx < 0 || factor_idx >= band_) {
    return;
  }

  double pivot = pivot_row[static_cast<size_t>(band_ - 1)];
  double factor = local_matrix_[(static_cast<size_t>(i) * static_cast<size_t>(band_)) + factor_idx] / pivot;

  if (std::fabs(factor) <= eps) {
    return;
  }

  UpdateRowValues(i, k, factor, pivot_row);
  local_vector_[static_cast<size_t>(i)] -= factor * pivot_b;
}

void IlinAGaussianMethodMPI::UpdateRowValues(int i, int k, double factor, const std::vector<double> &pivot_row) {
  int global_row = row_start_ + i;

  for (int j = 0; j < band_; ++j) {
    int src_idx = j - (global_row - k);
    if (src_idx < 0 || src_idx >= band_) {
      continue;
    }

    local_matrix_[(static_cast<size_t>(i) * static_cast<size_t>(band_)) + j] -=
        factor * pivot_row[static_cast<size_t>(src_idx)];
  }
}

void IlinAGaussianMethodMPI::ProcessBackwardSubstitution() {
  std::vector<double> recv_matrix;
  std::vector<double> recv_vector;

  GatherAllData(recv_matrix, recv_vector);

  if (rank_ != 0) {
    MPI_Bcast(solution_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return;
  }

  std::vector<double> full_matrix(static_cast<size_t>(n_) * static_cast<size_t>(band_));
  std::vector<double> full_vector(static_cast<size_t>(n_));

  ReconstructFullMatrix(recv_matrix, recv_vector, full_matrix, full_vector);
  SolveBackwardSubstitution(full_matrix, full_vector);

  MPI_Bcast(solution_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void IlinAGaussianMethodMPI::GatherAllData(std::vector<double> &recv_matrix, std::vector<double> &recv_vector) {
  if (rank_ == 0) {
    recv_matrix.resize(static_cast<size_t>(n_) * static_cast<size_t>(band_));
    recv_vector.resize(static_cast<size_t>(n_));
  }

  std::vector<int> recv_counts(static_cast<size_t>(size_));
  std::vector<int> displs(static_cast<size_t>(size_));
  std::vector<int> vec_counts(static_cast<size_t>(size_));
  std::vector<int> vec_displs(static_cast<size_t>(size_));

  for (int i = 0; i < size_; ++i) {
    int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
    recv_counts[static_cast<size_t>(i)] = rows_for_i * band_;
    vec_counts[static_cast<size_t>(i)] = rows_for_i;

    if (i == 0) {
      displs[0] = 0;
      vec_displs[0] = 0;
    } else {
      displs[static_cast<size_t>(i)] = displs[static_cast<size_t>(i - 1)] + recv_counts[static_cast<size_t>(i - 1)];
      vec_displs[static_cast<size_t>(i)] =
          vec_displs[static_cast<size_t>(i - 1)] + vec_counts[static_cast<size_t>(i - 1)];
    }
  }

  MPI_Gatherv(local_matrix_.data(), local_rows_ * band_, MPI_DOUBLE, recv_matrix.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(local_vector_.data(), local_rows_, MPI_DOUBLE, recv_vector.data(), vec_counts.data(), vec_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void IlinAGaussianMethodMPI::ReconstructFullMatrix(const std::vector<double> &recv_matrix,
                                                   const std::vector<double> &recv_vector,
                                                   std::vector<double> &full_matrix,
                                                   std::vector<double> &full_vector) const {
  std::vector<int> displs(static_cast<size_t>(size_));
  std::vector<int> vec_displs(static_cast<size_t>(size_));

  for (int i = 0; i < size_; ++i) {
    if (i == 0) {
      displs[0] = 0;
      vec_displs[0] = 0;
    } else {
      int prev_rows = (i - 1 < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
      displs[static_cast<size_t>(i)] = displs[static_cast<size_t>(i - 1)] + (prev_rows * band_);
      vec_displs[static_cast<size_t>(i)] = vec_displs[static_cast<size_t>(i - 1)] + prev_rows;
    }
  }

  for (int i = 0; i < size_; ++i) {
    int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
    int start_row = vec_displs[static_cast<size_t>(i)];

    for (int j = 0; j < rows_for_i; ++j) {
      int global_row = start_row + j;
      const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(displs[static_cast<size_t>(i)]) +
                                    (static_cast<std::ptrdiff_t>(j) * static_cast<std::ptrdiff_t>(band_));
      std::copy(&recv_matrix[static_cast<size_t>(offset)],
                &recv_matrix[static_cast<size_t>(offset) + static_cast<size_t>(band_)],
                &full_matrix[(static_cast<size_t>(global_row) * static_cast<size_t>(band_))]);
      full_vector[static_cast<size_t>(global_row)] =
          recv_vector[static_cast<size_t>(vec_displs[static_cast<size_t>(i)] + static_cast<size_t>(j))];
    }
  }
}

void IlinAGaussianMethodMPI::SolveBackwardSubstitution(const std::vector<double> &full_matrix,
                                                       const std::vector<double> &full_vector) {
  const double eps = 1e-12;

  for (int i = n_ - 1; i >= 0; --i) {
    double sum = 0.0;

    for (int j = i + 1; j < std::min(n_, i + band_); ++j) {
      int idx = band_ - 1 + (j - i);
      if (idx >= band_) {
        continue;
      }

      sum +=
          full_matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_)) + idx] * solution_[static_cast<size_t>(j)];
    }

    int diag_idx = band_ - 1;
    double diag = full_matrix[(static_cast<size_t>(i) * static_cast<size_t>(band_)) + diag_idx];

    if (std::fabs(diag) > eps) {
      solution_[static_cast<size_t>(i)] = (full_vector[static_cast<size_t>(i)] - sum) / diag;
    } else {
      solution_[static_cast<size_t>(i)] = 0.0;
    }
  }

  for (int i = 0; i < n_; ++i) {
    if (!std::isfinite(solution_[static_cast<size_t>(i)])) {
      solution_[static_cast<size_t>(i)] = 0.0;
    }
  }
}

bool IlinAGaussianMethodMPI::PostProcessingImpl() {
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank_ == 0) {
    GetOutput() = solution_;
  }

  MPI_Bcast(solution_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
