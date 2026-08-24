#!/bin/bash
cmake -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
-DCMAKE_FIND_DEBUG_MODE=OFF -DPROFILE_BUILD_PHASES:BOOL=ON --no-warn-unused-cli \
-S /home/sean/GpuHnsw -B /home/sean/GpuHnsw/build/Debug -G "Unix Makefiles"

cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
-DCMAKE_FIND_DEBUG_MODE=OFF -DPROFILE_BUILD_PHASES:BOOL=ON --no-warn-unused-cli \
-S /home/sean/GpuHnsw -B /home/sean/GpuHnsw/build/Release "Unix Makefiles"

# -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ \

cmake --build /home/sean/GpuHnsw/build/Release --config Release --target all -j 12
cmake --build /home/sean/GpuHnsw/build/Debug --config Debug --target all -j 12