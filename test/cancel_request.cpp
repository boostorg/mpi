#include <mpi.h>

#include <iostream>
#include <future>

#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/optional/optional_io.hpp>

using namespace std::literals::chrono_literals;
namespace mpi = boost::mpi;

void async_cancel(boost::mpi::request request)
{
  std::this_thread::sleep_for(1s);
  std::cout << "Before MPI_Cancel" << std::endl;
  request.cancel();
  std::cout << "After MPI_Cancel" << std::endl;
}


struct data
{
  int i;
};

template <typename Archive>
void serialize(Archive& ar, data& t, const unsigned int version)
{
  ar & t.i;
}

int main(int argc, char* argv[])
{
  mpi::environment env(mpi::threading::level::multiple);
  if (env.thread_level() >= mpi::threading::level::multiple) {
    std::cout << "Got necessary threading level.\n";
  } else {
    std::cerr << "Could not get required threading level.\n";
    return 0; // it's ok to fail
  }
  mpi::communicator world;
  if (world.rank() == 0)
  {
    //int buffer; // WORKS
    data buffer;  // FAILS
    auto request = world.irecv(0, 0, buffer);
    
    auto res = std::async(std::launch::async, &async_cancel, request);

    std::cout << "Before MPI_Wait" << std::endl;

#if defined(BOOST_MPI_USE_IMPROBE)
    boost::optional<mpi::status> status;
    while (request.active()) {
      std::cout << "Request still active.\n";
      status = request.test();
    }
    std::cout << "Canceled ? " << (status && status->cancelled()) << '\n';
#else
    request.wait();
#endif
    std::cout << "After MPI_Wait " << std::endl;
  }
  else
    std::this_thread::sleep_for(2s);

  return 0;
}
