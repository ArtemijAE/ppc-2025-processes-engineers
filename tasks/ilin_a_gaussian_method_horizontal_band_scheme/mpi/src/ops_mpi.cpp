#include "ilin_a_gaussian_method_horizontal_band_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace ilin_a_gaussian_method_horizontal_band_scheme {

IlinAGaussianMethodMPI::IlinAGaussianMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool IlinAGaussianMethodMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  if (input.size() < 4) {
    return false;
  }

  int size = static_cast<int>(input[0]);
  int band_width = static_cast<int>(input[1]);

  if (size <= 0 || band_width <= 0 || band_width > size) {
    return false;
  }

  size_t expected_count = 2 + size * band_width + size;
  if (input.size() != expected_count) {
    return false;
  }

  return true;
}

bool IlinAGaussianMethodMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count_);

  if (!ReadInputData(GetInput())) {
    return false;
  }

  solution_.resize(data_.size, 0.0);
  GetOutput() = std::vector<double>(data_.size, 0.0);

  return true;
}

bool IlinAGaussianMethodMPI::RunImpl() {
  DistributeData();
  GaussianEliminationMPI();
  BackSubstitutionMPI();
  GatherResults();

  if (rank_ == 0) {
    GetOutput() = solution_;
  }

  return true;
}

bool IlinAGaussianMethodMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    const int n = data_.size;
    const int m = data_.band_width;

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      int start_col = std::max(0, i - m + 1);
      int end_col = std::min(n - 1, i + m - 1);

      for (int j = start_col; j <= end_col; ++j) {
        int band_index = (j - i + m - 1);
        sum += data_.matrix[i * m + band_index] * solution_[j];
      }

      if (std::fabs(sum - data_.vector[i]) > 1e-6) {
        return false;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IlinAGaussianMethodMPI::ReadInputData(const std::vector<double> &input) {
  data_.size = static_cast<int>(input[0]);
  data_.band_width = static_cast<int>(input[1]);

  int matrix_elements = data_.size * data_.band_width;
  int vector_elements = data_.size;

  data_.matrix.resize(matrix_elements);
  data_.vector.resize(vector_elements);

  std::copy(input.begin() + 2, input.begin() + 2 + matrix_elements, data_.matrix.begin());

  std::copy(input.begin() + 2 + matrix_elements, input.end(), data_.vector.begin());

  return true;
}

void IlinAGaussianMethodMPI::DistributeData() {
  const int n = data_.size;
  const int m = data_.band_width;

  int rows_per_proc = n / proc_count_;
  int remainder = n % proc_count_;

  row_start_ = rank_ * rows_per_proc + std::min(rank_, remainder);
  row_end_ = row_start_ + rows_per_proc + (rank_ < remainder ? 1 : 0) - 1;
  local_rows_ = row_end_ - row_start_ + 1;

  std::vector<int> send_counts(proc_count_);
  std::vector<int> displs(proc_count_);

  for (int proc = 0; proc < proc_count_; ++proc) {
    int proc_rows = rows_per_proc + (proc < remainder ? 1 : 0);
    send_counts[proc] = proc_rows * m;
    displs[proc] = (proc == 0) ? 0 : (displs[proc - 1] + send_counts[proc - 1]);
  }

  local_matrix_.resize(local_rows_ * m);
  MPI_Scatterv(data_.matrix.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_matrix_.data(),
               local_rows_ * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<int> vec_counts(proc_count_);
  std::vector<int> vec_displs(proc_count_);

  for (int proc = 0; proc < proc_count_; ++proc) {
    int proc_rows = rows_per_proc + (proc < remainder ? 1 : 0);
    vec_counts[proc] = proc_rows;
    vec_displs[proc] = (proc == 0) ? 0 : (vec_displs[proc - 1] + vec_counts[proc - 1]);
  }

  local_vector_.resize(local_rows_);
  MPI_Scatterv(data_.vector.data(), vec_counts.data(), vec_displs.data(), MPI_DOUBLE, local_vector_.data(), local_rows_,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  local_solution_.resize(local_rows_);
}

void IlinAGaussianMethodMPI::GaussianEliminationMPI() {
  const int n = data_.size;
  const int m = data_.band_width;

  for (int k = 0; k < n; ++k) {
    int proc_with_k = k / (n / proc_count_ + (n % proc_count_ > 0 ? 1 : 0));
    if (proc_with_k >= proc_count_) {
      proc_with_k = proc_count_ - 1;
    }

    double max_val = 0.0;
    int max_row = k;

    if (rank_ == proc_with_k) {
      int local_k = k - row_start_;
      int band_index_kk = (k - k + m - 1);
      max_val = std::fabs(local_matrix_[local_k * m + band_index_kk]);
      max_row = k;

      for (int i = local_k; i < local_rows_ && (row_start_ + i) < std::min(n, row_start_ + local_rows_); ++i) {
        int global_i = row_start_ + i;
        if (global_i <= std::min(n - 1, k + m - 1)) {
          int band_index = (k - global_i + m - 1);
          if (band_index >= 0) {
            double val = std::fabs(local_matrix_[i * m + band_index]);
            if (val > max_val) {
              max_val = val;
              max_row = global_i;
            }
          }
        }
      }
    }

    struct {
      double value;
      int index;
    } local_max, global_max;

    local_max.value = max_val;
    local_max.index = max_row;

    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    max_row = global_max.index;
    int proc_with_max = max_row / (n / proc_count_ + (n % proc_count_ > 0 ? 1 : 0));

    if (max_row != k) {
      std::vector<double> row_k(m, 0.0);
      double vec_k = 0.0;

      if (rank_ == proc_with_k) {
        int local_k = k - row_start_;
        std::copy(local_matrix_.begin() + local_k * m, local_matrix_.begin() + local_k * m + m, row_k.begin());
        vec_k = local_vector_[local_k];
      }

      std::vector<double> row_max(m, 0.0);
      double vec_max = 0.0;

      if (rank_ == proc_with_max) {
        int local_max_row = max_row - row_start_;
        std::copy(local_matrix_.begin() + local_max_row * m, local_matrix_.begin() + local_max_row * m + m,
                  row_max.begin());
        vec_max = local_vector_[local_max_row];
      }

      MPI_Bcast(row_k.data(), m, MPI_DOUBLE, proc_with_k, MPI_COMM_WORLD);
      MPI_Bcast(&vec_k, 1, MPI_DOUBLE, proc_with_k, MPI_COMM_WORLD);
      MPI_Bcast(row_max.data(), m, MPI_DOUBLE, proc_with_max, MPI_COMM_WORLD);
      MPI_Bcast(&vec_max, 1, MPI_DOUBLE, proc_with_max, MPI_COMM_WORLD);

      if (rank_ == proc_with_k) {
        int local_k = k - row_start_;
        std::copy(row_max.begin(), row_max.end(), local_matrix_.begin() + local_k * m);
        local_vector_[local_k] = vec_max;
      }

      if (rank_ == proc_with_max) {
        int local_max_row = max_row - row_start_;
        std::copy(row_k.begin(), row_k.end(), local_matrix_.begin() + local_max_row * m);
        local_vector_[local_max_row] = vec_k;
      }
    }

    std::vector<double> pivot_row(m, 0.0);
    double pivot_vec = 0.0;

    if (rank_ == proc_with_k) {
      int local_k = k - row_start_;
      std::copy(local_matrix_.begin() + local_k * m, local_matrix_.begin() + local_k * m + m, pivot_row.begin());
      pivot_vec = local_vector_[local_k];
    }

    MPI_Bcast(pivot_row.data(), m, MPI_DOUBLE, proc_with_k, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_vec, 1, MPI_DOUBLE, proc_with_k, MPI_COMM_WORLD);

    int band_index_kk = (k - k + m - 1);
    double pivot_diag = pivot_row[band_index_kk];

    if (std::fabs(pivot_diag) < 1e-12) {
      continue;
    }

    for (int i = 0; i < local_rows_; ++i) {
      int global_i = row_start_ + i;

      if (global_i > k && global_i <= std::min(n - 1, k + m - 1)) {
        int band_index_ik = (k - global_i + m - 1);
        if (band_index_ik < 0 || band_index_ik >= m) {
          continue;
        }

        double factor = local_matrix_[i * m + band_index_ik] / pivot_diag;

        for (int j = k; j <= std::min(n - 1, k + m - 1); ++j) {
          int band_index_ij = (j - global_i + m - 1);
          int band_index_kj = (j - k + m - 1);

          if (band_index_ij >= 0 && band_index_ij < m && band_index_kj >= 0 && band_index_kj < m) {
            local_matrix_[i * m + band_index_ij] -= factor * pivot_row[band_index_kj];
          }
        }

        local_vector_[i] -= factor * pivot_vec;
      }
    }
  }
}

void IlinAGaussianMethodMPI::BackSubstitutionMPI() {
  const int n = data_.size;
  const int m = data_.band_width;

  std::vector<double> global_solution(n, 0.0);

  for (int i = n - 1; i >= 0; --i) {
    int proc_with_i = i / (n / proc_count_ + (n % proc_count_ > 0 ? 1 : 0));

    double diag_element = 0.0;
    double rhs = 0.0;

    if (rank_ == proc_with_i) {
      int local_i = i - row_start_;
      int band_index_ii = (i - i + m - 1);
      diag_element = local_matrix_[local_i * m + band_index_ii];
      rhs = local_vector_[local_i];
    }

    MPI_Bcast(&diag_element, 1, MPI_DOUBLE, proc_with_i, MPI_COMM_WORLD);
    MPI_Bcast(&rhs, 1, MPI_DOUBLE, proc_with_i, MPI_COMM_WORLD);

    double sum = 0.0;

    for (int j = 0; j < local_rows_; ++j) {
      int global_j = row_start_ + j;
      if (global_j > i && global_j <= std::min(n - 1, i + m - 1)) {
        int band_index = (global_j - i + m - 1);
        if (band_index >= 0 && band_index < m) {
          sum += local_matrix_[j * m + band_index] * global_solution[global_j];
        }
      }
    }

    double total_sum = 0.0;
    MPI_Allreduce(&sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (std::fabs(diag_element) > 1e-12) {
      global_solution[i] = (rhs - total_sum) / diag_element;
    } else {
      global_solution[i] = 0.0;
    }

    MPI_Bcast(&global_solution[i], 1, MPI_DOUBLE, proc_with_i, MPI_COMM_WORLD);
  }

  for (int i = 0; i < local_rows_; ++i) {
    int global_i = row_start_ + i;
    local_solution_[i] = global_solution[global_i];
  }
}

void IlinAGaussianMethodMPI::GatherResults() {
  const int n = data_.size;

  std::vector<int> recv_counts(proc_count_);
  std::vector<int> displs(proc_count_);

  int rows_per_proc = n / proc_count_;
  int remainder = n % proc_count_;

  for (int proc = 0; proc < proc_count_; ++proc) {
    recv_counts[proc] = rows_per_proc + (proc < remainder ? 1 : 0);
    displs[proc] = (proc == 0) ? 0 : (displs[proc - 1] + recv_counts[proc - 1]);
  }

  MPI_Gatherv(local_solution_.data(), local_rows_, MPI_DOUBLE, solution_.data(), recv_counts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

}  // namespace ilin_a_gaussian_method_horizontal_band_scheme
