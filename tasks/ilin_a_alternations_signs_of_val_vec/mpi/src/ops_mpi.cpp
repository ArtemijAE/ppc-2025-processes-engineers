#include "ilin_a_alternations_signs_of_val_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "ilin_a_alternations_signs_of_val_vec/common/include/common.hpp"

namespace ilin_a_alternations_signs_of_val_vec {

namespace {
constexpr int kRootRank = 0;
constexpr int kMinDataSize = 2;
}  // namespace

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
  const size_t segment_size = segment.size();

  if (segment_size < kMinDataSize) {
    return count;
  }

  for (size_t index = 0; index < segment_size - 1; ++index) {
    const bool is_negative_current = segment[index] < 0;
    const bool is_negative_next = segment[index + 1] < 0;
    if (is_negative_current != is_negative_next) {
      ++count;
    }
  }
  return count;
}

BoundaryInfo IlinAAlternationsSignsOfValVecMPI::GatherEdgeValues(const std::vector<int> &segment) {
  BoundaryInfo info;

  const int left_val = segment.empty() ? 0 : segment.front();
  const int right_val = segment.empty() ? 0 : segment.back();

  int total_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

  const size_t edges_size = static_cast<size_t>(2) * static_cast<size_t>(total_processes);
  info.all_edges.resize(edges_size);

  MPI_Gather(&left_val, 1, MPI_INT, info.all_edges.data(), 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
  MPI_Gather(&right_val, 1, MPI_INT, info.all_edges.data() + static_cast<size_t>(total_processes), 1, MPI_INT,
             kRootRank, MPI_COMM_WORLD);

  return info;
}

int IlinAAlternationsSignsOfValVecMPI::CountEdgeAlternations(const BoundaryInfo &edges, const int total_processes) {
  int count = 0;

  if (total_processes <= 1) {
    return count;
  }

  for (int process_index = 0; process_index < total_processes - 1; ++process_index) {
    const size_t right_index = static_cast<size_t>(total_processes) + static_cast<size_t>(process_index);
    const size_t left_index = static_cast<size_t>(process_index) + 1;
    const int right_edge = edges.all_edges[right_index];
    const int left_edge = edges.all_edges[left_index];
    const bool is_negative_right = right_edge < 0;
    const bool is_negative_left = left_edge < 0;
    if (is_negative_right != is_negative_left) {
      ++count;
    }
  }
  return count;
}

void IlinAAlternationsSignsOfValVecMPI::CalculateDistribution(const int data_size, const int world_size,
                                                              std::vector<int> &counts, std::vector<int> &offsets) {
  const int base_size = data_size / world_size;
  const int remainder = data_size % world_size;
  int current_offset = 0;

  for (int process_index = 0; process_index < world_size; ++process_index) {
    counts[process_index] = base_size + (process_index < remainder ? 1 : 0);
    offsets[process_index] = current_offset;
    current_offset += counts[process_index];
  }
}

void IlinAAlternationsSignsOfValVecMPI::DistributeData(const std::vector<int> &global_data,
                                                       std::vector<int> &local_data, const int world_rank,
                                                       const int world_size) {
  if (world_rank == kRootRank) {
    std::vector<int> counts(world_size);
    std::vector<int> offsets(world_size);
    CalculateDistribution(static_cast<int>(global_data.size()), world_size, counts, offsets);

    if (counts[kRootRank] > 0) {
      const auto start_iterator = global_data.begin() + static_cast<ptrdiff_t>(offsets[kRootRank]);
      const auto end_iterator = start_iterator + static_cast<ptrdiff_t>(counts[kRootRank]);
      std::copy(start_iterator, end_iterator, local_data.begin());
    }

    for (int process_index = 0; process_index < world_size; ++process_index) {
      if (process_index == kRootRank) {
        continue;
      }

      const int send_size = counts[process_index];
      if (send_size > 0) {
        const int *send_data = global_data.data() + offsets[process_index];
        MPI_Send(send_data, send_size, MPI_INT, process_index, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (!local_data.empty()) {
      MPI_Recv(local_data.data(), static_cast<int>(local_data.size()), MPI_INT, kRootRank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
}

bool IlinAAlternationsSignsOfValVecMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::vector<int> &input_data = GetInput();
  int data_size = static_cast<int>(input_data.size());
  MPI_Bcast(&data_size, 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
  if (data_size < kMinDataSize) {
    if (world_rank == kRootRank) {
      GetOutput() = 0;
    }
    return true;
  }
  std::vector<int> counts(world_size);
  std::vector<int> offsets(world_size);
  if (world_rank == kRootRank) {
    CalculateDistribution(data_size, world_size, counts, offsets);
  }
  MPI_Bcast(counts.data(), world_size, MPI_INT, kRootRank, MPI_COMM_WORLD);
  MPI_Bcast(offsets.data(), world_size, MPI_INT, kRootRank, MPI_COMM_WORLD);
  const int local_size = counts[world_rank];
  std::vector<int> local_data(static_cast<size_t>(local_size));
  DistributeData(input_data, local_data, world_rank, world_size);
  const int local_changes = CountLocalSignChanges(local_data);
  BoundaryInfo edges = GatherEdgeValues(local_data);
  int total_changes = 0;
  MPI_Reduce(&local_changes, &total_changes, 1, MPI_INT, MPI_SUM, kRootRank, MPI_COMM_WORLD);
  if (world_rank == kRootRank) {
    total_changes += CountEdgeAlternations(edges, world_size);
    GetOutput() = total_changes;
  }

  return true;
}

bool IlinAAlternationsSignsOfValVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ilin_a_alternations_signs_of_val_vec
