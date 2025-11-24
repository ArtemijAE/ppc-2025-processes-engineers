#include "ilin_a_alternations_signs_of_val_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

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

int IlinAAlternationsSignsOfValVecMPI::CountLocalSignChanges(const std::vector<int> &segment) {
  int count = 0;
  if (segment.size() < 2) {
    return count;
  }

  for (size_t i = 0; i < segment.size() - 1; ++i) {
    if ((segment[i] < 0) != (segment[i + 1] < 0)) {
      count++;
    }
  }
  return count;
}

BoundaryInfo IlinAAlternationsSignsOfValVecMPI::GatherEdgeValues(const std::vector<int> &segment) {
  BoundaryInfo info;
  int left_val = segment.empty() ? 0 : segment.front();
  int right_val = segment.empty() ? 0 : segment.back();

  int total_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

  info.all_edges.resize(2 * total_processes);
  MPI_Gather(&left_val, 1, MPI_INT, info.all_edges.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&right_val, 1, MPI_INT, info.all_edges.data() + total_processes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return info;
}

int IlinAAlternationsSignsOfValVecMPI::CountEdgeAlternations(const BoundaryInfo &edges, int total_processes) {
  int count = 0;
  for (int i = 0; i < total_processes - 1; ++i) {
    if ((edges.all_edges[total_processes + i] < 0) != (edges.all_edges[i + 1] < 0)) {
      count++;
    }
  }
  return count;
}

bool IlinAAlternationsSignsOfValVecMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::vector<int> &input = GetInput();
  int data_size = static_cast<int>(input.size());

  if (data_size < 2) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base = data_size / size;
  int rem = data_size % size;
  int local_size = base + (rank < rem ? 1 : 0);
  std::vector<int> local_data(local_size);

  if (rank == 0) {
    int offset = local_size;
    for (int i = 1; i < size; ++i) {
      int proc_size = base + (i < rem ? 1 : 0);
      MPI_Send(input.data() + offset, proc_size, MPI_INT, i, 0, MPI_COMM_WORLD);
      offset += proc_size;
    }
    std::copy(input.begin(), input.begin() + local_size, local_data.begin());
  } else {
    MPI_Recv(local_data.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_changes = CountLocalSignChanges(local_data);
  BoundaryInfo edges = GatherEdgeValues(local_data);

  int total_changes = 0;
  MPI_Reduce(&local_changes, &total_changes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    total_changes += CountEdgeAlternations(edges, size);
    GetOutput() = total_changes;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IlinAAlternationsSignsOfValVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_alternations_signs_of_val_vec
