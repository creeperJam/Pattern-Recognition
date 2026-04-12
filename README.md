# Pattern-Recognition

This project is a high-performance pattern recognition implementation designed to analyze multichannel time-series data. It identifies the best matching position for multiple queries across five data channels using the SAD (Sum of Absolute Differences) metric.

The core objective of this project is to implement, optimize, and compare two different parallel computing paradigms:

- **CPU Parallelism**: Utilizing modern C++ and OpenMP for multi-threading.

- **GPU Mass Parallelism**: Utilizing NVIDIA CUDA for extreme throughput.

By providing a side-by-side implementation, this project serves as a practical benchmark between CPU-bound multithreading and GPU-bound parallel accelerators.

## Building the project
### Requirements

#### C++
To build the C++ version you will need a compiler that supports **OpenMP** and the **C++20** standard. During our testing we used GCC 15.2.0.

*Note: If you are using a compiler different from the one originally used for this project  (e.g., switching between GCC and MSVC), ensure that the compiler-specific flags in the `CMakeLists.txt` are adjusted accordingly (for example, the optimization flag `-O3` used in GCC corresponds to `/O2` in MSVC).*

#### CUDA
To build the CUDA version you will need the following things:

- **C++ Compiler**: the compiler used during our tests was MSVC 19.44.35223.0 (since on Windows). If on Linux make sure to use the GCC compiler. Clangd should also work for both OSs, but we did not test them and can't guarantee it works.
- **Nvidia Toolkit**: we used version 13.0.88, so anything newer should work. Ensure the toolkit (`nvcc`) is properly added to your system's `Path`.
- **CMake**: we used version 4.0.2 during the tests, so we reccomend using that or any newer. But if not possible, version 3.18 is the other reccomended version. *Note: using versions different from the used by us may require some changes to the CMakeLists based on the chosen version*. 
- **CUDA drivers**: make sure the installed drivers are compatible with the chosen toolkit.
- **HARDWARE**: this is the most important. During our testing we used an RTX 4060 for laptops. Our code has `constexpr` that define values optimized for this specific card, if yours is different (more importantly, **older**), make sure to adjust the `BLOCK_SIZE, QUERY_LENGTH AND NUM_QUERIES` to more appropriate values. The main problem with older cards would be the lower size of constant and shared memory.

As said for the C++ version, make sure to change any and all compiler flags to the correct ones, if using different compilers.

*NOTE: in the CMakeLists we specificed the Compute Capability to 8.9. This corresponds to the Ada Lovelace architecture (e.g. RTX 4000). Not changing this while having a card that doesn't support the 8.9 will give errors.*

### Building
Once all the requirements are met the building is straight forward:

+ Go into the folder of the chosen version (cd `path/to/Pattern_Recognition_version)
+ Run `mkdir build` and `cd build`
+ Run `cmake .. ` and after it finishes `cmake --build . --config Release`.

If everything was done correctly and nothing is missing, the executable can be found in the build folder.