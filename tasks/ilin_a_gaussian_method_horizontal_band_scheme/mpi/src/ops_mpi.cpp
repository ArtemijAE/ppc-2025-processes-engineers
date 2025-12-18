#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ilin_a_gaussian_method_horizontal_band_scheme {

IlinAGaussianMethodMPI::IlinAGaussianMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool IlinAGaussianMethodMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  if (rank_ == 0) {
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

  return true;
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
    row_start_ = remainder_ * (rows_per_proc_ + 1) + ((rank_ - remainder_) * rows_per_proc_);
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
    std::copy(data_.matrix.begin() + global_row * band_, data_.matrix.begin() + (global_row + 1) * band_,
              local_matrix_.begin() + static_cast<size_t>(i) * static_cast<size_t>(band_));
    local_vector_[static_cast<size_t>(i)] = data_.vector[static_cast<size_t>(global_row)];
  }

  solution_.resize(static_cast<size_t>(n_), 0.0);
  pivot_row_buf_.resize(static_cast<size_t>(band_));
}

bool IlinAGaussianMethodMPI::RunImpl() {
  const double eps = 1e-12;
  pivot_row_buf_.assign(band_, 0.0);
  double pivot_b = 0.0;

  for (int k = 0; k < n_; ++k) {
    int global_max_row = -1;
    double global_max_val = 0.0;
    std::vector<double> pivot_row(band_, 0.0);
    int pivot_owner = -1;

    FindGlobalPivot(k, global_max_row, global_max_val, pivot_row, pivot_b, pivot_owner);

    if (global_max_val < eps) {
      solution_[static_cast<size_t>(k)] = 0.0;
      continue;
    }

    SwapRowsIfNeeded(k, global_max_row, pivot_owner);
    EliminateRows(k, pivot_row, pivot_b);
  }

  BackSubstitution();

  GetOutput() = solution_;
  return true;
}

void IlinAGaussianMethodMPI::FindGlobalPivot(int k, int &global_max_row, double &global_max_val,
                                             std::vector<double> &pivot_row, double &pivot_b, int &pivot_owner) const {
  double local_max = 0.0;
  int local_max_global_row = -1;

  for (int i = 0; i < local_rows_; ++i) {
    int global_row = row_start_ + i;
    if (global_row >= k) {
      int diag_idx = band_ - 1 - (global_row - k);
      if (diag_idx >= 0 && diag_idx < band_) {
        double val = std::fabs(local_matrix_[static_cast<size_t>(i) * static_cast<size_t>(band_) + diag_idx]);
        if (val > local_max) {
          local_max = val;
          local_max_global_row = global_row;
        }
      }
    }
  }

  struct {
    double max_val;
    int max_row;
  } local_data{.max_val = local_max, .max_row = local_max_global_row}, global_data{.max_val = 0.0, .max_row = -1};

  MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  global_max_row = global_data.max_row;
  global_max_val = global_data.max_val;

  if (global_max_row >= 0) {
    if (global_max_row < remainder_ * (rows_per_proc_ + 1)) {
      pivot_owner = global_max_row / (rows_per_proc_ + 1);
    } else {
      pivot_owner = remainder_ + ((global_max_row - remainder_ * (rows_per_proc_ + 1)) / rows_per_proc_);
    }
  }

  if (pivot_owner == rank_) {
    int local_pivot_idx = -1;
    for (int i = 0; i < local_rows_; ++i) {
      if (row_start_ + i == global_max_row) {
        local_pivot_idx = i;
        break;
      }
    }

    if (local_pivot_idx >= 0) {
      std::copy(&local_matrix_[static_cast<size_t>(local_pivot_idx) * static_cast<size_t>(band_)],
                &local_matrix_[static_cast<size_t>(local_pivot_idx) * static_cast<size_t>(band_)] + band_,
                pivot_row.begin());
      pivot_b = local_vector_[static_cast<size_t>(local_pivot_idx)];
    }
  }

  MPI_Bcast(pivot_row.data(), band_, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
  MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
}

void IlinAGaussianMethodMPI::SwapRowsIfNeeded(int k, int global_max_row, int pivot_owner) {
  if (global_max_row != k) {
    int k_owner = -1;
    if (k < remainder_ * (rows_per_proc_ + 1)) {
      k_owner = k / (rows_per_proc_ + 1);
    } else {
      k_owner = remainder_ + ((k - remainder_ * (rows_per_proc_ + 1)) / rows_per_proc_);
    }

    if (k_owner == rank_) {
      int k_local_idx = -1;
      for (int i = 0; i < local_rows_; ++i) {
        if (row_start_ + i == k) {
          k_local_idx = i;
          break;
        }
      }

      if (k_local_idx >= 0) {
        if (pivot_owner == rank_) {
          int pivot_local_idx = -1;
          for (int i = 0; i < local_rows_; ++i) {
            if (row_start_ + i == global_max_row) {
              pivot_local_idx = i;
              break;
            }
          }

          if (pivot_local_idx >= 0) {
            std::swap_ranges(&local_matrix_[static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_)],
                             &local_matrix_[static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_)] + band_,
                             &local_matrix_[static_cast<size_t>(pivot_local_idx) * static_cast<size_t>(band_)]);
            std::swap(local_vector_[static_cast<size_t>(k_local_idx)],
                      local_vector_[static_cast<size_t>(pivot_local_idx)]);
          }
        } else {
          std::vector<double> temp_row(band_);
          double temp_b = 0.0;
          MPI_Status status;

          MPI_Recv(temp_row.data(), band_, MPI_DOUBLE, pivot_owner, 0, MPI_COMM_WORLD, &status);
          MPI_Recv(&temp_b, 1, MPI_DOUBLE, pivot_owner, 1, MPI_COMM_WORLD, &status);

          std::ranges::copy(temp_row, &local_matrix_[static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_)]);
          local_vector_[static_cast<size_t>(k_local_idx)] = temp_b;

          MPI_Send(&local_matrix_[static_cast<size_t>(k_local_idx) * static_cast<size_t>(band_)], band_, MPI_DOUBLE,
                   pivot_owner, 2, MPI_COMM_WORLD);
          MPI_Send(&local_vector_[static_cast<size_t>(k_local_idx)], 1, MPI_DOUBLE, pivot_owner, 3, MPI_COMM_WORLD);
        }
      }
    } else if (pivot_owner == rank_) {
      int pivot_local_idx = -1;
      for (int i = 0; i < local_rows_; ++i) {
        if (row_start_ + i == global_max_row) {
          pivot_local_idx = i;
          break;
        }
      }

      if (pivot_local_idx >= 0) {
        MPI_Send(&local_matrix_[static_cast<size_t>(pivot_local_idx) * static_cast<size_t>(band_)], band_, MPI_DOUBLE,
                 k_owner, 0, MPI_COMM_WORLD);
        MPI_Send(&local_vector_[static_cast<size_t>(pivot_local_idx)], 1, MPI_DOUBLE, k_owner, 1, MPI_COMM_WORLD);

        std::vector<double> temp_row(band_);
        double temp_b = 0.0;
        MPI_Status status;

        MPI_Recv(temp_row.data(), band_, MPI_DOUBLE, k_owner, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&temp_b, 1, MPI_DOUBLE, k_owner, 3, MPI_COMM_WORLD, &status);

        std::ranges::copy(temp_row, &local_matrix_[static_cast<size_t>(pivot_local_idx) * static_cast<size_t>(band_)]);
        local_vector_[static_cast<size_t>(pivot_local_idx)] = temp_b;
      }
    }
  }
}

void IlinAGaussianMethodMPI::EliminateRows(int k, const std::vector<double> &pivot_row, double pivot_b) {
  const double eps = 1e-12;
  double pivot = pivot_row[static_cast<size_t>(band_ - 1)];
  if (std::fabs(pivot) < eps) {
    return;
  }

  for (int i = 0; i < local_rows_; ++i) {
    int global_row = row_start_ + i;
    if (global_row > k) {
      int factor_idx = band_ - 1 - (global_row - k);
      if (factor_idx >= 0 && factor_idx < band_) {
        double factor = local_matrix_[static_cast<size_t>(i) * static_cast<size_t>(band_) + factor_idx] / pivot;

        if (std::fabs(factor) > eps) {
          for (int j = 0; j < band_; ++j) {
            int src_idx = j - (global_row - k);
            if (src_idx >= 0 && src_idx < band_) {
              local_matrix_[static_cast<size_t>(i) * static_cast<size_t>(band_) + j] -=
                  factor * pivot_row[static_cast<size_t>(src_idx)];
            }
          }

          local_vector_[static_cast<size_t>(i)] -= factor * pivot_b;
        }
      }
    }
  }
}

void IlinAGaussianMethodMPI::GatherResults() {
  std::vector<double> recv_matrix;
  std::vector<double> recv_vector;

  if (rank_ == 0) {
    recv_matrix.resize(static_cast<size_t>(n_) * static_cast<size_t>(band_));
    recv_vector.resize(static_cast<size_t>(n_));
  }

  std::vector<int> recv_counts(static_cast<size_t>(size_));
  std::vector<int> displs(static_cast<size_t>(size_));

  for (int i = 0; i < size_; ++i) {
    int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
    recv_counts[static_cast<size_t>(i)] = rows_for_i * band_;
    displs[static_cast<size_t>(i)] =
        (i == 0) ? 0 : displs[static_cast<size_t>(i - 1)] + recv_counts[static_cast<size_t>(i - 1)];
  }

  MPI_Gatherv(local_matrix_.data(), local_rows_ * band_, MPI_DOUBLE, recv_matrix.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<int> vec_counts(static_cast<size_t>(size_));
  std::vector<int> vec_displs(static_cast<size_t>(size_));

  for (int i = 0; i < size_; ++i) {
    int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
    vec_counts[static_cast<size_t>(i)] = rows_for_i;
    vec_displs[static_cast<size_t>(i)] =
        (i == 0) ? 0 : vec_displs[static_cast<size_t>(i - 1)] + vec_counts[static_cast<size_t>(i - 1)];
  }

  MPI_Gatherv(local_vector_.data(), local_rows_, MPI_DOUBLE, recv_vector.data(), vec_counts.data(), vec_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void IlinAGaussianMethodMPI::BackSubstitution() {
  const double eps = 1e-12;

  std::vector<double> recv_matrix;
  std::vector<double> recv_vector;

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
    displs[static_cast<size_t>(i)] =
        (i == 0) ? 0 : displs[static_cast<size_t>(i - 1)] + recv_counts[static_cast<size_t>(i - 1)];
    vec_displs[static_cast<size_t>(i)] =
        (i == 0) ? 0 : vec_displs[static_cast<size_t>(i - 1)] + vec_counts[static_cast<size_t>(i - 1)];
  }

  MPI_Gatherv(local_matrix_.data(), local_rows_ * band_, MPI_DOUBLE, recv_matrix.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(local_vector_.data(), local_rows_, MPI_DOUBLE, recv_vector.data(), vec_counts.data(), vec_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<double> full_matrix(static_cast<size_t>(n_) * static_cast<size_t>(band_));
    std::vector<double> full_vector(static_cast<size_t>(n_));

    for (int i = 0; i < size_; ++i) {
      int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
      int start_row = vec_displs[static_cast<size_t>(i)];

      for (int j = 0; j < rows_for_i; ++j) {
        int global_row = start_row + j;
        std::copy(&recv_matrix[static_cast<size_t>(displs[static_cast<size_t>(i)] + (j * band_))],
                  &recv_matrix[static_cast<size_t>(displs[static_cast<size_t>(i)] + (j * band_))] + band_,
                  &full_matrix[static_cast<size_t>(global_row) * static_cast<size_t>(band_)]);
        full_vector[static_cast<size_t>(global_row)] =
            recv_vector[static_cast<size_t>(vec_displs[static_cast<size_t>(i)] + j)];
      }
    }

    for (int i = n_ - 1; i >= 0; --i) {
      double sum = 0.0;

      for (int j = i + 1; j < std::min(n_, i + band_); ++j) {
        int idx = band_ - 1 + (j - i);
        if (idx < band_) {
          sum += full_matrix[static_cast<size_t>(i) * static_cast<size_t>(band_) + idx] *
                 solution_[static_cast<size_t>(j)];
        }
      }

      int diag_idx = band_ - 1;
      double diag = full_matrix[static_cast<size_t>(i) * static_cast<size_t>(band_) + diag_idx];

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

  MPI_Bcast(solution_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool IlinAGaussianMethodMPI::PostProcessingImpl() {
  MPI_Barrier(MPI_COMM_WORLD);

  double local_max_error = 0.0;

  for (int i = row_start_; i < row_end_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n_; ++j) {
      int band_idx = (j - i + band_ - 1);
      if (band_idx >= 0 && band_idx < band_) {
        double matrix_elem = data_.matrix[static_cast<size_t>(i) * static_cast<size_t>(band_) + band_idx];
        sum += matrix_elem * solution_[static_cast<size_t>(j)];
      }
    }

    double vector_elem = data_.vector[static_cast<size_t>(i)];
    double error = std::fabs(sum - vector_elem);
    local_max_error = std::max(error, local_max_error);
  }

  double global_max_error = 0.0;
  MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_max_error < 1e-6;
}

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
