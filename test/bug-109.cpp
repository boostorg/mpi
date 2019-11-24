#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/mpi.hpp>

#include <cassert>

namespace mpi = boost::mpi;

struct Range {
  uint64_t start;
  uint64_t interval;
};

struct Comm_range {
  Range range;
  uint64_t id;
};

struct Comm_found {
  std::vector<uint64_t> found;
  uint64_t id;
};

struct Found_from_range {
  Range range;
  Comm_found comm_found;
};

namespace boost {
  namespace serialization {
    template<class Archive> void serialize(Archive & ar, Range & r, const unsigned int /*version*/){
      ar & r.start;
      ar & r.interval;
    }

    template<class Archive> void serialize(Archive & ar, Comm_range & r, const unsigned int /*version*/){
      ar & r.range;
      ar & r.id;
    }
    
    template<class Archive> void serialize(Archive & ar, Comm_found & r, const unsigned int /*version*/){
      ar & r.found;
      ar & r.id;
    }

  }
  namespace mpi {
    template <> struct is_mpi_datatype<Range> : mpl::true_ { };
    template <> struct is_mpi_datatype<Comm_range> : mpl::true_ { };
  }
}



void append_result(std::vector<Range> &cost, uint64_t num){

  if(cost.empty() || (num - (cost.back().start + cost.back().interval) > 2)){
    //insert at the end
    Range tmp = {num, 0};
    cost.push_back(tmp);
  }else{
    cost.back().interval += 2;
  }
}


std::vector<uint64_t> work_function(const Range range){
  std::vector<uint64_t> found;
  
  for(uint64_t current = range.start; current <= range.start + range.interval; current += 2){
    if(current % 71 == 0){
      found.push_back(current);
    }
  }

  return found;
}


int main(){

  mpi::environment mpi_env;
  mpi::communicator mpi_world;

  const uint32_t stopping_objective = 100;

  if(mpi_world.rank() == 0){

    std::vector<Range> unknown;
    unknown.push_back({3, 256});
    unknown.push_back({289, 476});

    std::vector<Range> future;

    const int number_slaves = mpi_world.size() - 1;
    const int buffer_size = 10;
    assert(number_slaves > 0);

    uint32_t raw_objective = 5;
    broadcast(mpi_world, raw_objective, 0);

    std::vector<mpi::request> pending_isends(buffer_size);
    std::vector<Comm_range> pending_isend_buffer(buffer_size);

    std::vector<Found_from_range> unmapped_results;

    auto it_unknown = unknown.cbegin();

    uint64_t workunit_counter;
    uint64_t lowest_id_not_found = 0;

    for(workunit_counter = 0; workunit_counter < buffer_size && it_unknown != unknown.cend(); ++workunit_counter, ++it_unknown){
      pending_isend_buffer[workunit_counter] = Comm_range({Range({it_unknown->start, it_unknown->interval}), workunit_counter});
      pending_isends[workunit_counter] = mpi_world.isend( (workunit_counter % number_slaves) + 1, workunit_counter, pending_isend_buffer[workunit_counter]);
    }

    Comm_found result;
    mpi::request pending_recv_message = mpi_world.irecv(mpi::any_source, mpi::any_tag, result);
    boost::optional<mpi::status> recv_test_result = boost::none;

    bool found_lowest_id = false;

    std::vector<int> terminated_slaves;

    while(lowest_id_not_found < workunit_counter){
      if(!recv_test_result){
        mpi_world.probe(mpi::any_source, mpi::any_tag);
        recv_test_result = pending_recv_message.test();
      }

      while(recv_test_result){
        
        const int32_t buffer_no = recv_test_result->tag();
        const int32_t slave_no = recv_test_result->source();

        pending_isends[buffer_no].wait();
        assert(pending_isend_buffer[buffer_no].id == result.id);
        
        const Range sent_range = pending_isend_buffer[buffer_no].range;

        if(std::find(terminated_slaves.begin(), terminated_slaves.end(), slave_no) == terminated_slaves.end()){
          pending_isend_buffer[buffer_no] = Comm_range({Range({ 0ul, 0ul}), 0xFFFFFFFFFFFFFFFF});
          pending_isends[buffer_no] = mpi_world.isend(slave_no, buffer_no, pending_isend_buffer[buffer_no]);
          terminated_slaves.push_back(slave_no);
        }
        

        {
          Found_from_range new_item = Found_from_range({sent_range, result});
          unmapped_results.insert(
                                  std::upper_bound(unmapped_results.begin(),
                                                   unmapped_results.end(),
                                                   new_item,
                                                   [](Found_from_range a, Found_from_range b){ return (a.comm_found.id < b.comm_found.id); } ),
                                  new_item
                                  );
        }

        found_lowest_id |= (result.id == lowest_id_not_found);
        pending_recv_message = mpi_world.irecv(mpi::any_source, mpi::any_tag, result);

        recv_test_result = pending_recv_message.test();
        
      }

      if(found_lowest_id){
        
        uint64_t old_id = unmapped_results.begin()->comm_found.id;
        auto it_unmapped = unmapped_results.begin();
        for(; it_unmapped != unmapped_results.end() && it_unmapped->comm_found.id - old_id <= 1; ++it_unmapped){

          auto itf = it_unmapped->comm_found.found.begin();
          for(uint64_t current = it_unmapped->range.start;
              current <= it_unmapped->range.start + it_unmapped->range.interval; current += 2ul){
            if(itf < it_unmapped->comm_found.found.end() && current == *itf){
              ++itf;
            }else{
              
              append_result(future, current);

            }
          }
          
          old_id = it_unmapped->comm_found.id;
        }
        lowest_id_not_found = old_id + 1u;
        unmapped_results.erase(unmapped_results.begin(), it_unmapped);

        found_lowest_id = false;

      }

    }

    if(unknown.size() == 0){
      for(int slave_no = 1; slave_no <= number_slaves; ++slave_no){
        pending_isend_buffer[slave_no] = Comm_range({Range({ 0ul, 0ul}), 0xFFFFFFFFFFFFFFFF});
        pending_isends[slave_no] = mpi_world.isend(slave_no, slave_no, pending_isend_buffer[slave_no]);
      }
    }
    pending_recv_message.cancel();
    //pending_recv_message.wait();
    //std::cout << "cancelled: " << pending_recv_message.wait().cancelled() << std::endl;
    wait_all(pending_isends.begin(), pending_isends.end());
    /*for(auto current = pending_isends.begin(); current != pending_isends.end(); ++current){
      if(! current->active()){
      current->wait();
      }
      }*/
    

    unknown.swap(future);
    future.clear();
    assert(unmapped_results.size() == 0);
    
    uint32_t send_objective = stopping_objective;
    broadcast(mpi_world, send_objective, 0);

    for(size_t i = 0; i < std::min(unknown.size(), 10ul); ++i){
      std::cout << unknown[i].start << " until " << unknown[i].start + unknown[i].interval << std::endl;
    }

  }else{

    uint32_t raw_objective;
    broadcast(mpi_world, raw_objective, 0);

    while(raw_objective != stopping_objective){
      bool done = false;

      Comm_found pending_result;
      mpi::request pending_message;

      while(!done){

        Comm_range workunit;
        mpi::status status_recv = mpi_world.recv(0, mpi::any_tag, workunit);
        const int tag = status_recv.tag();
        
        if(workunit.id == 0xFFFFFFFFFFFFFFFF){
          done = true;
        }else{

          std::vector<uint64_t> found = work_function(workunit.range);

          pending_message.wait();
          pending_result = Comm_found({found, workunit.id});
          pending_message = mpi_world.isend(0, tag, pending_result);

        }
      }

      pending_message.wait();

      broadcast(mpi_world, raw_objective, 0);

    }
  }

  return 0;

}
