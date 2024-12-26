#include <papi.h>
#include <iostream>
#include <stdexcept>

class CPUCacheProfiler {
 public:
  CPUCacheProfiler() : eventSet(PAPI_NULL) {
    // Initialize the PAPI library
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
      throw std::runtime_error("PAPI library initialization failed");
    }

    // Create the event set
    int ret = PAPI_create_eventset(&eventSet);
    if (ret != PAPI_OK) {
      throw std::runtime_error("PAPI event set creation failed");
    }

    ret = PAPI_add_event(eventSet, PAPI_L1_DCA);  // L1 Data Cache Accesses
    if (ret != PAPI_OK) {
      throw std::runtime_error(
          "PAPI add event failed for L1 Data Cache Accesses");
    }

    // Add events to the event set
    ret = PAPI_add_event(eventSet, PAPI_L1_DCM);  // L1 Data Cache Misses
    if (ret != PAPI_OK) {
      throw std::runtime_error(
          "PAPI add event failed for L1 Data Cache Misses");
    }

    ret = PAPI_add_event(eventSet, PAPI_L2_DCH);  // L2 data cache hit
    if (ret != PAPI_OK) {
      throw std::runtime_error("PAPI add event failed for L2 Data Cache Hit");
    }

    ret = PAPI_add_event(eventSet, PAPI_L2_DCM);  // L2 data cache miss
    if (ret != PAPI_OK) {
      throw std::runtime_error("PAPI add event failed for L2 Data Cache Miss");
    }
  }
  ~CPUCacheProfiler() {
    // Cleanup the event set
    if (eventSet != PAPI_NULL) {
      PAPI_cleanup_eventset(eventSet);
    }
  }

  void start() {
    int ret = PAPI_start(eventSet);
    if (ret != PAPI_OK) {
      throw std::runtime_error("PAPI start failed");
    }
  }

  void stop() {
    int ret = PAPI_stop(eventSet, values);
    if (ret != PAPI_OK) {
      throw std::runtime_error("PAPI stop failed");
    }
  }

  void report() const {
    long long L1_access = values[0];
    long long L1_miss = values[1];
    long long L2_hit = values[2];
    long long L2_miss = values[3];
    // float missRatio = (accesses != 0) ? static_cast<float>(misses) /
    // static_cast<float>(accesses) : 0.0f;

    std::cout << "L1 Data Cache Accesses:   " << L1_access << std::endl;
    std::cout << "L1 Data Cache Misses:     " << L1_miss << std::endl;
    std::cout << "L2 Data Cache Hits:       " << L2_hit << std::endl;
    std::cout << "L2 Data Cache Misses:     " << L2_miss << std::endl;
  }

 private:
  int eventSet;
  long long values[4];
};