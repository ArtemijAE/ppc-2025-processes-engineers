#include "ilin_a_alternations_signs_of_val_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

IlinAAlternationsSignsOfValVecMPI::IlinAAlternationsSignsOfValVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool IlinAAlternationsSignsOfValVecMPI::ValidationImpl() {
  return GetOutput() == 0;
}

bool IlinAAlternationsSignsOfValVecMPI::PreProcessingImpl() {
  return true;
}

bool IlinAAlternationsSignsOfValVecMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::vector<int> &global_vec = GetInput();
  int global_size = static_cast<int>(global_vec.size());

  if (global_size < 2) {
    if (world_rank == 0) {
      GetOutput() = 0;  // +2
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = global_size / world_size;
  int remainder = global_size % world_size;
  int extra = (world_rank < remainder) ? 1 : 0;
  std::vector<int> local_vec(local_size + extra);

  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);

  if (world_rank == 0) {  // +1
    for (int i = 0; i < world_size; ++i) {
      counts[i] = local_size + (i < remainder ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < world_size; ++i) {  // +2
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Scatterv(const_cast<int *>(global_vec.data()), counts.data(), displs.data(), MPI_INT, local_vec.data(),
               static_cast<int>(local_vec.size()), MPI_INT, 0, MPI_COMM_WORLD);

  int local_alternations = 0;
  size_t loop_limit = (local_vec.size() >= 2) ? local_vec.size() - 1 : 0;

  for (size_t i = 0; i < loop_limit; ++i) {
    bool sign1 = local_vec[i] < 0;
    bool sign2 = local_vec[i + 1] < 0;
    if (sign1 != sign2) {
      local_alternations++;
    }
  }

  int left_boundary = local_vec.empty() ? 0 : local_vec.front();
  int right_boundary = local_vec.empty() ? 0 : local_vec.back();

  std::vector<int> boundaries(static_cast<size_t>(2) * world_size);
  MPI_Gather(&left_boundary, 1, MPI_INT, boundaries.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&right_boundary, 1, MPI_INT, boundaries.data() + world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_alternations = 0;
  MPI_Reduce(&local_alternations, &total_alternations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {                        // +1
    for (int i = 0; i < world_size - 1; ++i) {  // +2
      int right_of_i = boundaries[world_size + i];
      int left_of_next = boundaries[i + 1];
      bool sign1 = right_of_i < 0;
      bool sign2 = left_of_next < 0;
      if (sign1 != sign2) {
        total_alternations++;  // +1
      }
    }
    GetOutput() = total_alternations;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IlinAAlternationsSignsOfValVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_alternations_signs_of_val_vec
