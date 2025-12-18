#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ilin_a_gaussian_method_horizontal_band_scheme {

IlinAGaussianMethodMPI::IlinAGaussianMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
  rank_ = 0;
  size_ = 1;
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

    size_t expected_count = 2 + size * band_width + size;
    return input.size() == expected_count;
  }

  return true;
}

bool IlinAGaussianMethodMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

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
    row_start_ = remainder_ * (rows_per_proc_ + 1) + (rank_ - remainder_) * rows_per_proc_;
  }
  row_end_ = row_start_ + local_rows_;

  if (rank_ == 0) {
    const auto &input = GetInput();
    int mat_size = n_ * band_;
    data_.matrix.resize(mat_size);
    data_.vector.resize(n_);

    std::copy(input.begin() + 2, input.begin() + 2 + mat_size, data_.matrix.begin());
    std::copy(input.begin() + 2 + mat_size, input.end(), data_.vector.begin());
  } else {
    data_.matrix.resize(n_ * band_);
    data_.vector.resize(n_);
  }

  MPI_Bcast(data_.matrix.data(), n_ * band_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(data_.vector.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  local_matrix_.resize(local_rows_ * band_);
  local_vector_.resize(local_rows_);

  for (int i = 0; i < local_rows_; ++i) {
    int global_row = row_start_ + i;
    std::copy(data_.matrix.begin() + global_row * band_, data_.matrix.begin() + (global_row + 1) * band_,
              local_matrix_.begin() + i * band_);
    local_vector_[i] = data_.vector[global_row];
  }

  solution_.resize(n_, 0.0);

  return true;
}

bool IlinAGaussianMethodMPI::RunImpl() {
  std::vector<double> pivot_row(band_, 0.0);
  double pivot_b = 0.0;
  const double EPS = 1e-12;

  for (int k = 0; k < n_; ++k) {
    double local_max = 0.0;
    int local_max_global_row = -1;

    for (int i = 0; i < local_rows_; ++i) {
      int global_row = row_start_ + i;
      if (global_row >= k) {
        int diag_idx = band_ - 1 - (global_row - k);
        if (diag_idx >= 0 && diag_idx < band_) {
          double val = std::fabs(local_matrix_[i * band_ + diag_idx]);
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
    } local_data{local_max, local_max_global_row}, global_data{0.0, -1};

    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    int global_max_row = global_data.max_row;
    double global_max_val = global_data.max_val;

    if (global_max_val < EPS) {
      solution_[k] = 0.0;
      continue;
    }

    int pivot_owner = -1;
    if (global_max_row >= 0) {
      if (global_max_row < remainder_ * (rows_per_proc_ + 1)) {
        pivot_owner = global_max_row / (rows_per_proc_ + 1);
      } else {
        pivot_owner = remainder_ + (global_max_row - remainder_ * (rows_per_proc_ + 1)) / rows_per_proc_;
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
        std::copy(&local_matrix_[local_pivot_idx * band_], &local_matrix_[local_pivot_idx * band_] + band_,
                  pivot_row.begin());
        pivot_b = local_vector_[local_pivot_idx];
      }
    }

    MPI_Bcast(pivot_row.data(), band_, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

    if (global_max_row != k) {
      int k_owner = -1;
      if (k < remainder_ * (rows_per_proc_ + 1)) {
        k_owner = k / (rows_per_proc_ + 1);
      } else {
        k_owner = remainder_ + (k - remainder_ * (rows_per_proc_ + 1)) / rows_per_proc_;
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
              std::swap_ranges(&local_matrix_[k_local_idx * band_], &local_matrix_[k_local_idx * band_] + band_,
                               &local_matrix_[pivot_local_idx * band_]);
              std::swap(local_vector_[k_local_idx], local_vector_[pivot_local_idx]);
            }
          } else {
            std::vector<double> temp_row(band_);
            double temp_b;
            MPI_Status status;

            MPI_Recv(temp_row.data(), band_, MPI_DOUBLE, pivot_owner, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&temp_b, 1, MPI_DOUBLE, pivot_owner, 1, MPI_COMM_WORLD, &status);

            std::copy(temp_row.begin(), temp_row.end(), &local_matrix_[k_local_idx * band_]);
            local_vector_[k_local_idx] = temp_b;

            MPI_Send(&local_matrix_[k_local_idx * band_], band_, MPI_DOUBLE, pivot_owner, 2, MPI_COMM_WORLD);
            MPI_Send(&local_vector_[k_local_idx], 1, MPI_DOUBLE, pivot_owner, 3, MPI_COMM_WORLD);
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
          MPI_Send(&local_matrix_[pivot_local_idx * band_], band_, MPI_DOUBLE, k_owner, 0, MPI_COMM_WORLD);
          MPI_Send(&local_vector_[pivot_local_idx], 1, MPI_DOUBLE, k_owner, 1, MPI_COMM_WORLD);

          std::vector<double> temp_row(band_);
          double temp_b;
          MPI_Status status;

          MPI_Recv(temp_row.data(), band_, MPI_DOUBLE, k_owner, 2, MPI_COMM_WORLD, &status);
          MPI_Recv(&temp_b, 1, MPI_DOUBLE, k_owner, 3, MPI_COMM_WORLD, &status);

          std::copy(temp_row.begin(), temp_row.end(), &local_matrix_[pivot_local_idx * band_]);
          local_vector_[pivot_local_idx] = temp_b;
        }
      }
    }

    double pivot = pivot_row[band_ - 1];
    if (std::fabs(pivot) < EPS) {
      continue;
    }

    for (int i = 0; i < local_rows_; ++i) {
      int global_row = row_start_ + i;
      if (global_row > k) {
        int factor_idx = band_ - 1 - (global_row - k);
        if (factor_idx >= 0 && factor_idx < band_) {
          double factor = local_matrix_[i * band_ + factor_idx] / pivot;

          if (std::fabs(factor) > EPS) {
            for (int j = 0; j < band_; ++j) {
              int src_idx = j - (global_row - k);
              if (src_idx >= 0 && src_idx < band_) {
                local_matrix_[i * band_ + j] -= factor * pivot_row[src_idx];
              }
            }

            local_vector_[i] -= factor * pivot_b;
          }
        }
      }
    }
  }

  std::vector<double> recv_matrix;
  std::vector<double> recv_vector;

  if (rank_ == 0) {
    recv_matrix.resize(n_ * band_);
    recv_vector.resize(n_);
  }

  std::vector<int> recv_counts(size_);
  std::vector<int> displs(size_);

  for (int i = 0; i < size_; ++i) {
    int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
    recv_counts[i] = rows_for_i * band_;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recv_counts[i - 1];
  }

  MPI_Gatherv(local_matrix_.data(), local_rows_ * band_, MPI_DOUBLE, recv_matrix.data(), recv_counts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<int> vec_counts(size_);
  std::vector<int> vec_displs(size_);

  for (int i = 0; i < size_; ++i) {
    int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
    vec_counts[i] = rows_for_i;
    vec_displs[i] = (i == 0) ? 0 : vec_displs[i - 1] + vec_counts[i - 1];
  }

  MPI_Gatherv(local_vector_.data(), local_rows_, MPI_DOUBLE, recv_vector.data(), vec_counts.data(), vec_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<double> full_matrix(n_ * band_);
    std::vector<double> full_vector(n_);

    for (int i = 0; i < size_; ++i) {
      int rows_for_i = (i < remainder_) ? (rows_per_proc_ + 1) : rows_per_proc_;
      int start_row = vec_displs[i];

      for (int j = 0; j < rows_for_i; ++j) {
        int global_row = start_row + j;
        std::copy(&recv_matrix[displs[i] + j * band_], &recv_matrix[displs[i] + j * band_] + band_,
                  &full_matrix[global_row * band_]);
        full_vector[global_row] = recv_vector[vec_displs[i] + j];
      }
    }

    for (int i = n_ - 1; i >= 0; --i) {
      double sum = 0.0;

      for (int j = i + 1; j < std::min(n_, i + band_); ++j) {
        int idx = band_ - 1 + (j - i);
        if (idx < band_) {
          sum += full_matrix[i * band_ + idx] * solution_[j];
        }
      }

      int diag_idx = band_ - 1;
      double diag = full_matrix[i * band_ + diag_idx];

      if (std::fabs(diag) > EPS) {
        solution_[i] = (full_vector[i] - sum) / diag;
      } else {
        solution_[i] = 0.0;
      }
    }

    for (int i = 0; i < n_; ++i) {
      if (!std::isfinite(solution_[i])) {
        solution_[i] = 0.0;
      }
    }
  }

  MPI_Bcast(solution_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = solution_;

  return true;
}

bool IlinAGaussianMethodMPI::PostProcessingImpl() {
  MPI_Barrier(MPI_COMM_WORLD);

  double local_max_error = 0.0;

  for (int i = row_start_; i < row_end_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n_; ++j) {
      int band_idx = (j - i + band_ - 1);
      if (band_idx >= 0 && band_idx < band_) {
        double matrix_elem = data_.matrix[i * band_ + band_idx];
        sum += matrix_elem * solution_[j];
      }
    }

    double vector_elem = data_.vector[i];
    double error = std::fabs(sum - vector_elem);
    if (error > local_max_error) {
      local_max_error = error;
    }
  }

  double global_max_error;
  MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_max_error < 1e-6;
}

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
