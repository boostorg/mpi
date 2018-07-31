#include <vector>
#include <iostream>
#include <iterator>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

namespace mpi = boost::mpi;

int main(int argc, char **argv)
{
    bool iswap = argc > 1 && argv[1] == std::string("iswap");

    mpi::environment env(argc, argv);
    mpi::communicator comm_world;
    auto rank = comm_world.rank();

    if (rank == 0) {
        std::vector<int> data;
        if (iswap) {
            auto req = comm_world.irecv(1, 0, data);
            req.wait();
        } else {
            comm_world.recv(1, 0, data);
        }
        std::cout << "Process 0 received:" << std::endl;
        std::copy(std::begin(data), std::end(data), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;

    } else if (rank == 1) {
        std::vector<int> vec = {1, 2, 3, 4, 5};
        if (iswap) {
            comm_world.send(0, 0, vec);
        } else {
            auto req = comm_world.isend(0, 0, vec);
            req.wait();
        }
    } 
}
