#include <iostream>
#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/collectives.hpp>

struct huge {
  std::vector<unsigned char> data;
  huge() : data(2ull << 30ull, 0) { }
  
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar & data;
  }
};

int main()
{
  boost::mpi::environment env;
  boost::mpi::communicator world;
  
  huge a{};
  
  std::cout << world.rank() << " huge created "  << std::endl;
  world.barrier();
  
  if (world.rank() == 0) {
      std::vector<huge> all;
      boost::mpi::gather(world, a, all, 0);
  } else  {
    boost::mpi::gather(world, a, 0);
  }
  
  return 0;
}
