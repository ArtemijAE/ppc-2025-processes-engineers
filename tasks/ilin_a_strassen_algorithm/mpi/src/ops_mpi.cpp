#include "ilin_a_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "ilin_a_strassen_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace ilin_a_strassen_algorithm {

IlinAStrassenAlgorithmMPI::IlinAStrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().size = 0;
  GetOutput().C.clear();

  world_size_ = 0;
  world_rank_ = 0;
  original_size_ = 0;
  padded_size_ = 0;
  threshold_ = 64;
}

bool IlinAStrassenAlgorithmMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetOutput().size = 0;
  GetOutput().C.clear();

  if (rank == 0) {
    const auto &input = GetInput();

    int validation_result = 1;
    int size_to_bcast = input.size;

    if (input.size <= 0) {
      validation_result = 0;
    }
    if (input.A.size() != static_cast<size_t>(input.size * input.size)) {
      validation_result = 0;
    }
    if (input.B.size() != static_cast<size_t>(input.size * input.size)) {
      validation_result = 0;
    }

    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (validation_result) {
      MPI_Bcast(&size_to_bcast, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return true;
    } else {
      int error_size = -1;
      MPI_Bcast(&error_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return false;
    }
  } else {
    int validation_result;
    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (validation_result) {
      int size;
      MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return size > 0;
    } else {
      int error_size;
      MPI_Bcast(&error_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return false;
    }
  }
}

bool IlinAStrassenAlgorithmMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

  const auto &input = GetInput();

  if (world_rank_ == 0) {
    original_size_ = input.size;

    int power = 1;
    while (power < original_size_) {
      power *= 2;
    }
    padded_size_ = power;
  }

  MPI_Bcast(&original_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&padded_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void IlinAStrassenAlgorithmMPI::addMatrix(const std::vector<double> &A, const std::vector<double> &B,
                                          std::vector<double> &C, int n) {
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] + B[i];
  }
}

void IlinAStrassenAlgorithmMPI::subtractMatrix(const std::vector<double> &A, const std::vector<double> &B,
                                               std::vector<double> &C, int n) {
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] - B[i];
  }
}

void IlinAStrassenAlgorithmMPI::splitMatrix(const std::vector<double> &A, std::vector<double> &A11,
                                            std::vector<double> &A12, std::vector<double> &A21,
                                            std::vector<double> &A22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A11[i * half + j] = A[i * n + j];
      A12[i * half + j] = A[i * n + (j + half)];
      A21[i * half + j] = A[(i + half) * n + j];
      A22[i * half + j] = A[(i + half) * n + (j + half)];
    }
  }
}

void IlinAStrassenAlgorithmMPI::joinMatrix(std::vector<double> &A, const std::vector<double> &A11,
                                           const std::vector<double> &A12, const std::vector<double> &A21,
                                           const std::vector<double> &A22, int n) {
  int half = n / 2;
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      A[i * n + j] = A11[i * half + j];
      A[i * n + (j + half)] = A12[i * half + j];
      A[(i + half) * n + j] = A21[i * half + j];
      A[(i + half) * n + (j + half)] = A22[i * half + j];
    }
  }
}

std::vector<double> IlinAStrassenAlgorithmMPI::naiveMultiplySeq(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
  std::vector<double> C(n * n, 0.0);

  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double aik = A[i * n + k];
      for (int j = 0; j < n; ++j) {
        C[i * n + j] += aik * B[k * n + j];
      }
    }
  }

  return C;
}

std::vector<double> IlinAStrassenAlgorithmMPI::strassenSequential(const std::vector<double> &A,
                                                                  const std::vector<double> &B, int n) {
  if (n <= threshold_) {
    return naiveMultiplySeq(A, B, n);
  }

  int half = n / 2;

  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  splitMatrix(A, A11, A12, A21, A22, n);
  splitMatrix(B, B11, B12, B21, B22, n);

  std::vector<double> P1, P2, P3, P4, P5, P6, P7;
  std::vector<double> temp1(half * half), temp2(half * half);

  addMatrix(A11, A22, temp1, half);
  addMatrix(B11, B22, temp2, half);
  P1 = strassenSequential(temp1, temp2, half);

  addMatrix(A21, A22, temp1, half);
  P2 = strassenSequential(temp1, B11, half);

  subtractMatrix(B12, B22, temp1, half);
  P3 = strassenSequential(A11, temp1, half);

  subtractMatrix(B21, B11, temp1, half);
  P4 = strassenSequential(A22, temp1, half);

  addMatrix(A11, A12, temp1, half);
  P5 = strassenSequential(temp1, B22, half);

  subtractMatrix(A21, A11, temp1, half);
  addMatrix(B11, B12, temp2, half);
  P6 = strassenSequential(temp1, temp2, half);

  subtractMatrix(A12, A22, temp1, half);
  addMatrix(B21, B22, temp2, half);
  P7 = strassenSequential(temp1, temp2, half);

  std::vector<double> C11(half * half), C12(half * half), C21(half * half), C22(half * half);

  addMatrix(P1, P4, C11, half);
  subtractMatrix(C11, P5, C11, half);
  addMatrix(C11, P7, C11, half);

  addMatrix(P3, P5, C12, half);

  addMatrix(P2, P4, C21, half);

  addMatrix(P1, P3, C22, half);
  subtractMatrix(C22, P2, C22, half);
  addMatrix(C22, P6, C22, half);

  std::vector<double> C(n * n);
  joinMatrix(C, C11, C12, C21, C22, n);

  return C;
}

std::vector<double> IlinAStrassenAlgorithmMPI::distributedNaiveMultiply(const std::vector<double> &A,
                                                                        const std::vector<double> &B, int n) {
  int rows_per_proc = n / world_size_;
  int remainder = n % world_size_;

  int local_rows = rows_per_proc + (world_rank_ < remainder ? 1 : 0);
  int start_row = 0;

  for (int i = 0; i < world_rank_; ++i) {
    int rows_for_i = rows_per_proc + (i < remainder ? 1 : 0);
    start_row += rows_for_i;
  }

  std::vector<double> local_result;
  if (local_rows > 0) {
    local_result.resize(local_rows * n, 0.0);

    for (int i = 0; i < local_rows; ++i) {
      int global_i = start_row + i;
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += A[global_i * n + k] * B[k * n + j];
        }
        local_result[i * n + j] = sum;
      }
    }
  }

  std::vector<double> final_result;

  if (world_rank_ == 0) {
    final_result.resize(n * n);
  }

  std::vector<int> recvcounts(world_size_);
  std::vector<int> displs(world_size_);

  if (world_rank_ == 0) {
    int offset = 0;
    for (int i = 0; i < world_size_; ++i) {
      int rows_for_i = rows_per_proc + (i < remainder ? 1 : 0);
      recvcounts[i] = rows_for_i * n;
      displs[i] = offset;
      offset += recvcounts[i];
    }
  }

  int send_count = static_cast<int>(local_result.size());
  const double *send_data = local_result.empty() ? nullptr : local_result.data();
  double *recv_data = world_rank_ == 0 ? final_result.data() : nullptr;
  int *recvcounts_ptr = world_rank_ == 0 ? recvcounts.data() : nullptr;
  int *displs_ptr = world_rank_ == 0 ? displs.data() : nullptr;

  MPI_Gatherv(send_data, send_count, MPI_DOUBLE, recv_data, recvcounts_ptr, displs_ptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank_ == 0) {
    std::vector<double> reordered_result(n * n);

    int offset = 0;
    for (int proc = 0; proc < world_size_; ++proc) {
      int rows_for_proc = rows_per_proc + (proc < remainder ? 1 : 0);
      int proc_start_row = 0;

      for (int i = 0; i < proc; ++i) {
        proc_start_row += rows_per_proc + (i < remainder ? 1 : 0);
      }

      for (int i = 0; i < rows_for_proc; ++i) {
        int global_row = proc_start_row + i;
        for (int j = 0; j < n; ++j) {
          reordered_result[global_row * n + j] = final_result[offset + i * n + j];
        }
      }
      offset += rows_for_proc * n;
    }

    return reordered_result;
  }

  return std::vector<double>();
}

std::vector<double> IlinAStrassenAlgorithmMPI::parallelStrassenRecursive(const std::vector<double> &A,
                                                                         const std::vector<double> &B, int n) {
  if (n <= threshold_) {
    return strassenSequential(A, B, n);
  }

  int half = n / 2;

  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  splitMatrix(A, A11, A12, A21, A22, n);
  splitMatrix(B, B11, B12, B21, B22, n);

  std::vector<double> P1, P2, P3, P4, P5, P6, P7;
  std::vector<double> temp1(half * half), temp2(half * half);

  std::vector<double> all_P1, all_P2, all_P3, all_P4, all_P5, all_P6, all_P7;

  int matrices_per_process = 7 / world_size_;
  int extra_matrices = 7 % world_size_;

  int start_matrix = 0;
  int end_matrix = 0;
  int matrices_to_compute = matrices_per_process;

  if (world_rank_ < extra_matrices) {
    matrices_to_compute++;
  }

  for (int i = 0; i < world_rank_; i++) {
    int matrices_for_i = matrices_per_process + (i < extra_matrices ? 1 : 0);
    start_matrix += matrices_for_i;
  }
  end_matrix = start_matrix + matrices_to_compute;

  for (int matrix_idx = start_matrix; matrix_idx < end_matrix; matrix_idx++) {
    switch (matrix_idx) {
      case 0:
        addMatrix(A11, A22, temp1, half);
        addMatrix(B11, B22, temp2, half);
        P1 = parallelStrassenRecursive(temp1, temp2, half);
        break;
      case 1:
        addMatrix(A21, A22, temp1, half);
        P2 = parallelStrassenRecursive(temp1, B11, half);
        break;
      case 2:
        subtractMatrix(B12, B22, temp1, half);
        P3 = parallelStrassenRecursive(A11, temp1, half);
        break;
      case 3:
        subtractMatrix(B21, B11, temp1, half);
        P4 = parallelStrassenRecursive(A22, temp1, half);
        break;
      case 4:
        addMatrix(A11, A12, temp1, half);
        P5 = parallelStrassenRecursive(temp1, B22, half);
        break;
      case 5:
        subtractMatrix(A21, A11, temp1, half);
        addMatrix(B11, B12, temp2, half);
        P6 = parallelStrassenRecursive(temp1, temp2, half);
        break;
      case 6:
        subtractMatrix(A12, A22, temp1, half);
        addMatrix(B21, B22, temp2, half);
        P7 = parallelStrassenRecursive(temp1, temp2, half);
        break;
    }
  }

  int matrix_size = half * half;

  std::vector<double> P1_buffer, P2_buffer, P3_buffer, P4_buffer, P5_buffer, P6_buffer, P7_buffer;

  if (P1.empty()) {
    P1.resize(matrix_size, 0.0);
  }
  if (P2.empty()) {
    P2.resize(matrix_size, 0.0);
  }
  if (P3.empty()) {
    P3.resize(matrix_size, 0.0);
  }
  if (P4.empty()) {
    P4.resize(matrix_size, 0.0);
  }
  if (P5.empty()) {
    P5.resize(matrix_size, 0.0);
  }
  if (P6.empty()) {
    P6.resize(matrix_size, 0.0);
  }
  if (P7.empty()) {
    P7.resize(matrix_size, 0.0);
  }

  P1_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P1.data(), matrix_size, MPI_DOUBLE, P1_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  P2_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P2.data(), matrix_size, MPI_DOUBLE, P2_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  P3_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P3.data(), matrix_size, MPI_DOUBLE, P3_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  P4_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P4.data(), matrix_size, MPI_DOUBLE, P4_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  P5_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P5.data(), matrix_size, MPI_DOUBLE, P5_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  P6_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P6.data(), matrix_size, MPI_DOUBLE, P6_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  P7_buffer.resize(matrix_size * world_size_);
  MPI_Allgather(P7.data(), matrix_size, MPI_DOUBLE, P7_buffer.data(), matrix_size, MPI_DOUBLE, MPI_COMM_WORLD);

  for (int proc = 0; proc < world_size_; proc++) {
    int offset = proc * matrix_size;

    bool P1_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P1_buffer[offset + i] != 0.0) {
        P1_has_data = true;
        break;
      }
    }
    if (P1_has_data && P1[0] == 0.0) {
      std::copy(P1_buffer.begin() + offset, P1_buffer.begin() + offset + matrix_size, P1.begin());
    }

    bool P2_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P2_buffer[offset + i] != 0.0) {
        P2_has_data = true;
        break;
      }
    }
    if (P2_has_data && P2[0] == 0.0) {
      std::copy(P2_buffer.begin() + offset, P2_buffer.begin() + offset + matrix_size, P2.begin());
    }

    bool P3_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P3_buffer[offset + i] != 0.0) {
        P3_has_data = true;
        break;
      }
    }
    if (P3_has_data && P3[0] == 0.0) {
      std::copy(P3_buffer.begin() + offset, P3_buffer.begin() + offset + matrix_size, P3.begin());
    }

    bool P4_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P4_buffer[offset + i] != 0.0) {
        P4_has_data = true;
        break;
      }
    }
    if (P4_has_data && P4[0] == 0.0) {
      std::copy(P4_buffer.begin() + offset, P4_buffer.begin() + offset + matrix_size, P4.begin());
    }

    bool P5_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P5_buffer[offset + i] != 0.0) {
        P5_has_data = true;
        break;
      }
    }
    if (P5_has_data && P5[0] == 0.0) {
      std::copy(P5_buffer.begin() + offset, P5_buffer.begin() + offset + matrix_size, P5.begin());
    }

    bool P6_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P6_buffer[offset + i] != 0.0) {
        P6_has_data = true;
        break;
      }
    }
    if (P6_has_data && P6[0] == 0.0) {
      std::copy(P6_buffer.begin() + offset, P6_buffer.begin() + offset + matrix_size, P6.begin());
    }

    bool P7_has_data = false;
    for (int i = 0; i < matrix_size; i++) {
      if (P7_buffer[offset + i] != 0.0) {
        P7_has_data = true;
        break;
      }
    }
    if (P7_has_data && P7[0] == 0.0) {
      std::copy(P7_buffer.begin() + offset, P7_buffer.begin() + offset + matrix_size, P7.begin());
    }
  }

  std::vector<double> C11(half * half), C12(half * half), C21(half * half), C22(half * half);

  addMatrix(P1, P4, C11, half);
  subtractMatrix(C11, P5, C11, half);
  addMatrix(C11, P7, C11, half);

  addMatrix(P3, P5, C12, half);

  addMatrix(P2, P4, C21, half);

  addMatrix(P1, P3, C22, half);
  subtractMatrix(C22, P2, C22, half);
  addMatrix(C22, P6, C22, half);

  std::vector<double> C(n * n);
  joinMatrix(C, C11, C12, C21, C22, n);

  return C;
}

std::vector<double> IlinAStrassenAlgorithmMPI::parallelStrassen(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
  return parallelStrassenRecursive(A, B, n);
}

std::vector<double> IlinAStrassenAlgorithmMPI::multiplyMatrices(const std::vector<double> &A,
                                                                const std::vector<double> &B, int n) {
  if (n <= threshold_) {
    return distributedNaiveMultiply(A, B, n);
  } else {
    return parallelStrassen(A, B, n);
  }
}

bool IlinAStrassenAlgorithmMPI::RunImpl() {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<double> A_full, B_full;

  if (world_rank_ == 0) {
    auto &input = GetInput();
    A_full = input.A;
    B_full = input.B;
  }

  std::vector<double> final_result;

  if (original_size_ <= threshold_) {
    if (world_rank_ != 0) {
      A_full.resize(original_size_ * original_size_);
      B_full.resize(original_size_ * original_size_);
    }

    MPI_Bcast(A_full.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_full.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    final_result = distributedNaiveMultiply(A_full, B_full, original_size_);
  } else {
    std::vector<double> A_padded(padded_size_ * padded_size_, 0.0);
    std::vector<double> B_padded(padded_size_ * padded_size_, 0.0);

    if (world_rank_ == 0) {
      for (int i = 0; i < original_size_; ++i) {
        for (int j = 0; j < original_size_; ++j) {
          A_padded[i * padded_size_ + j] = A_full[i * original_size_ + j];
          B_padded[i * padded_size_ + j] = B_full[i * original_size_ + j];
        }
      }
    }

    if (world_rank_ != 0) {
      A_padded.resize(padded_size_ * padded_size_);
      B_padded.resize(padded_size_ * padded_size_);
    }

    MPI_Bcast(A_padded.data(), padded_size_ * padded_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_padded.data(), padded_size_ * padded_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto C_padded = multiplyMatrices(A_padded, B_padded, padded_size_);

    if (world_rank_ == 0 && !C_padded.empty()) {
      final_result.resize(original_size_ * original_size_);
      for (int i = 0; i < original_size_; ++i) {
        for (int j = 0; j < original_size_; ++j) {
          final_result[i * original_size_ + j] = C_padded[i * padded_size_ + j];
        }
      }
    }
  }

  auto &output = GetOutput();

  if (world_rank_ == 0) {
    if (!final_result.empty()) {
      output.C = final_result;
      output.size = original_size_;

      MPI_Bcast(final_result.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    final_result.resize(original_size_ * original_size_);
    MPI_Bcast(final_result.data(), original_size_ * original_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    output.C = final_result;
    output.size = original_size_;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  return true;
}

bool IlinAStrassenAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_strassen_algorithm
