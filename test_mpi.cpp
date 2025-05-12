#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <mpi.h>
#include <cassert>
#include <utility>
#include <map>

constexpr static int NUM_ELEMENTS = 512;
constexpr static int NUM_LEVELS = 4;
constexpr static int NUM_STREAMS = 48;
constexpr bool USE_STREAMS = true;

MPI_Comm stream_comm;
std::vector<double> send_messages;
std::vector<double> recv_messages;

std::map<std::pair<int, int>, int> elem_pair_to_stream_idx;

void parse_neighbor_file(std::string filename, std::vector<std::vector<int>>& send_neighbors, std::vector<std::vector<int>>& recv_neighbors) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<int> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stoi(cell));
        }

	row.pop_back();

	if (row.size() > 0) {
	    int elem_number = row[0];
	    for (int i = 1; i < row.size() - 1; i++) {
		send_neighbors[elem_number].push_back(row[i]);
		recv_neighbors[row[i]].push_back(elem_number);
	    }
	}
    }

    file.close();
}

void init_elements_at_level(std::vector<int>* elements_at_level) {
    for (int i = 0; i < 64; i++) {
	elements_at_level[0].push_back(i);
    }
    for (int i = 64; i < 256; i++) {
	elements_at_level[1].push_back(i);
    }
    for (int i = 256; i < 448; i++) {
	elements_at_level[2].push_back(i);
    }
    for (int i = 448; i < 512; i++) {
	elements_at_level[3].push_back(i);
    }
}

int get_mpi_tag(int dst, int src) {
    return (dst << 14) | src;
}

void async_receive(int world_size, int rank, std::vector<MPI_Request>& recv_requests, std::map<std::pair<int, int>, int> elem_pair_to_recv_idx, const std::vector<int>& elem_to_stream_mapping) {
    recv_requests.resize(elem_pair_to_recv_idx.size());
    for (auto& [elem_pair, recv_idx] : elem_pair_to_recv_idx) {
	int src = elem_pair.first;
	int dst = elem_pair.second;
	// int send_stream_idx = 0;
        // int recv_stream_idx = 0;
        // int send_stream_idx = elem_to_stream_mapping[src];
        // int recv_stream_idx = elem_to_stream_mapping[dst];
        int send_stream_idx = elem_pair_to_stream_idx[{src, dst}];
        int recv_stream_idx = send_stream_idx;
	
	int mpi_tag = get_mpi_tag(dst, src);
	if (USE_STREAMS) {
	    MPIX_Stream_irecv(&recv_messages[recv_idx], 10000, MPI_DOUBLE,
			      src % world_size, mpi_tag,
			      stream_comm, send_stream_idx, recv_stream_idx,
			      &recv_requests[recv_idx]);
	} else {
	    MPI_Irecv(&recv_messages[recv_idx], 1, MPI_DOUBLE,
		      src % world_size, mpi_tag,
		      MPI_COMM_WORLD, &recv_requests[recv_idx]);
	}
    }
}

void main_loop(int world_size, int rank, std::vector<int>* elements_at_level,
	       const std::vector<std::vector<int>>& send_neighbors,
	       const std::vector<std::vector<int>>& recv_neighbors,
	       std::map<std::pair<int, int>, int>* elem_pair_to_recv_idx,
	       const std::vector<int>& elem_to_stream_mapping) {

    std::vector<std::vector<MPI_Request>> send_requests(NUM_ELEMENTS);
    std::vector<MPI_Request> recv_requests[NUM_LEVELS];

    int send_idx = 0;

    for (int level = 0; level < NUM_LEVELS; level++) {
	// Call async receives for elements in level + 1
	if (level < NUM_LEVELS - 1) {
	    async_receive(world_size, rank, recv_requests[level + 1], elem_pair_to_recv_idx[level + 1], elem_to_stream_mapping);
	}

	std::stringstream s1;
	s1 << "rank: " << rank << " level: " << level << " start wait." << std::endl;
	std::cout << s1.str();

	int num_wait = 0;
	while (num_wait < elem_pair_to_recv_idx[level].size()) {
	    int idx;
	    MPI_Waitany(recv_requests[level].size(), recv_requests[level].data(), &idx, MPI_STATUSES_IGNORE);
	    num_wait++;
	}

	std::stringstream s2;
	s2 << "rank: " << rank << " level: " << level << " end wait." << std::endl;
	std::cout << s2.str();

	for (int j = 0; j < elements_at_level[level].size(); j++) {
	    int elem = elements_at_level[level][j];
	    if (elem % world_size != rank) {
		continue;
	    }

	    // Send data from elem to neighbors
	    for (int neigh : send_neighbors[elem]) {
		if (neigh % world_size == rank) {
		    continue;
		}

	        int mpi_tag = get_mpi_tag(neigh, elem);
                // int send_stream_idx = 0;
                // int recv_stream_idx = 0;
	
                // int send_stream_idx = elem_to_stream_mapping[elem];
                // int recv_stream_idx = elem_to_stream_mapping[neigh];

                int send_stream_idx = elem_pair_to_stream_idx[{elem, neigh}];
                int recv_stream_idx = send_stream_idx;

		send_requests[elem].emplace_back();

		if (USE_STREAMS) {
		    MPIX_Stream_isend(&send_messages[send_idx++], 10000, MPI_DOUBLE,
				      neigh % world_size, mpi_tag,
				      stream_comm, send_stream_idx, recv_stream_idx,
				      &send_requests[elem][send_requests[elem].size() - 1]);
		} else {
		    MPI_Isend(&send_messages[send_idx++], 1, MPI_DOUBLE,
			      neigh % world_size, mpi_tag,
			      MPI_COMM_WORLD, &send_requests[elem][send_requests[elem].size() - 1]);
		}
	    }
	}
    }

    for (int level = 0; level < NUM_LEVELS; level++) {
	for (int j = 0; j < elements_at_level[level].size(); j++) {
	    int elem = elements_at_level[level][j];
	    if (elem % world_size != rank) {
		continue;
	    }

	    MPI_Waitall(send_requests[elem].size(), send_requests[elem].data(), MPI_STATUSES_IGNORE);
	}
    }
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided >= MPI_THREAD_MULTIPLE);

    std::string filename = "neigh.txt";
    std::vector<std::vector<int>> send_neighbors(NUM_ELEMENTS);
    std::vector<std::vector<int>> recv_neighbors(NUM_ELEMENTS);

    parse_neighbor_file(filename, send_neighbors, recv_neighbors);

    std::vector<int> elements_at_level[NUM_LEVELS];
    init_elements_at_level(elements_at_level);

    MPIX_Stream all_streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        MPIX_Stream_create(MPI_INFO_NULL, &all_streams[i]);
    }

    auto res = MPIX_Stream_comm_create_multiplex(MPI_COMM_WORLD, NUM_STREAMS, all_streams, &stream_comm);
    assert(res == MPI_SUCCESS);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    send_messages.resize(100000);
    recv_messages.resize(100000);

    std::map<std::pair<int, int>, int> elem_pair_to_recv_idx[NUM_LEVELS];
    // std::map<int, std::pair<int, int>> recv_idx_to_zoid_pair[NUM_LEVELS];

    std::vector<MPI_Request> recv_requests[NUM_LEVELS];

    for (int level = 0; level < NUM_LEVELS; level++) {
	int recv_idx = 0;
        for (int j = 0; j < elements_at_level[level].size(); j++) {
	    int elem = elements_at_level[level][j];
	    if (elem % world_size != rank) {
		continue;
	    }

	    auto& neighbors = recv_neighbors[elem];
	    for (int neigh : neighbors) {
		if (neigh % world_size == rank) {
		    continue;
		}
		// recv_idx_to_elem_pair[level][recv_idx] = {neigh, elem};
		elem_pair_to_recv_idx[level][{neigh, elem}] = recv_idx;
		recv_idx++;
	    }
	}

	recv_requests[level].resize(recv_idx, MPI_REQUEST_NULL);
    }

    std::vector<int> elem_to_stream_mapping(NUM_ELEMENTS, 0);
    
    int stream_idx = 0;
    
    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int j = 0; j < elements_at_level[level].size(); j++) {
	    int elem = elements_at_level[level][j];
	    if (elem % world_size != rank) {
		continue;
	    }

	    elem_to_stream_mapping[elem] = stream_idx;
	    stream_idx = (stream_idx + 1);
	}
    }

    stream_idx = 0;
    int total = 0;

    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int j = 0; j < elements_at_level[level].size(); j++) {
	    int elem = elements_at_level[level][j];
	    if (elem % world_size != rank) {
		continue;
	    }

	    auto& neighbors = send_neighbors[elem];
	    for (int neigh : neighbors) {
		if (neigh % world_size == rank) {
		    continue;
		}
	        elem_pair_to_stream_idx[{elem, neigh}] = stream_idx;
		assert(stream_idx < NUM_STREAMS);
	        stream_idx = (stream_idx + 1);
		total++;
	    }
	}
    }

    std::cout << "total: " << total << std::endl;

    std::vector<int> my_src;
    std::vector<int> my_dst;
    std::vector<int> my_stream_nums;

    for (auto& [k, v] : elem_pair_to_stream_idx) {
        my_src.push_back(k.first);
        my_dst.push_back(k.second);
        my_stream_nums.push_back(v);
    }

    std::vector<int> counts(world_size, 0);
    std::vector<int> displacements(world_size, 0);

    int my_count = my_src.size();
    MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_size = 0;
    for (int proc = 0; proc < world_size; proc++) {
        total_size += counts[proc];
    }

    displacements[0] = 0;
    for (int proc = 1; proc < world_size; proc++) {
        displacements[proc] = displacements[proc - 1] + counts[proc - 1];
    }

    std::vector<int> all_src;
    std::vector<int> all_dst;
    std::vector<int> all_comm_idx;
    all_src.resize(total_size);
    all_dst.resize(total_size);
    all_comm_idx.resize(total_size);

    MPI_Allgatherv(my_src.data(), counts[rank], MPI_INT, all_src.data(),
                   counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    MPI_Allgatherv(my_dst.data(), counts[rank], MPI_INT, all_dst.data(),
                   counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    MPI_Allgatherv(my_stream_nums.data(), counts[rank], MPI_INT, all_comm_idx.data(),
                   counts.data(), displacements.data(), MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < all_src.size(); i++) {
        int src = all_src[i];
        int dst = all_dst[i];
        int comm_idx = all_comm_idx[i];
        elem_pair_to_stream_idx[{src, dst}] = comm_idx;
    }

    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int j = 0; j < elements_at_level[level].size(); j++) {
	    int elem = elements_at_level[level][j];
	    if (elem % world_size != rank) {
		continue;
	    }

	    auto& neighbors = send_neighbors[elem];
	    for (int neigh : neighbors) {
		if (neigh % world_size == rank) {
		    continue;
		}
	    }
	}
    }
    

    MPI_Allreduce(MPI_IN_PLACE, elem_to_stream_mapping.data(), NUM_ELEMENTS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    main_loop(world_size, rank, elements_at_level, send_neighbors, recv_neighbors, elem_pair_to_recv_idx, elem_to_stream_mapping);

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "rank: " << rank << " done." << std::endl;

    MPI_Comm_free(&stream_comm);

    for (int i = 0; i < NUM_STREAMS; i++) {
        MPIX_Stream_free(&all_streams[i]);
    }

    return 0;
}

