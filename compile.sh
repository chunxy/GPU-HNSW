#!/bin/bash
cd "$(dirname "$0")"

mkdir -p ./build/Debug
mkdir -p ./build/Release

cmake -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
-DCMAKE_FIND_DEBUG_MODE=OFF -DPROFILE_BUILD_PHASES:BOOL=ON --no-warn-unused-cli \
-S ./ -B ./build/Debug -G "Unix Makefiles"

cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
-DCMAKE_FIND_DEBUG_MODE=OFF -DPROFILE_BUILD_PHASES:BOOL=OFF --no-warn-unused-cli \
-S ./ -B ./build/Release "Unix Makefiles"

# -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ \

cmake --build ./build/Release --config Release --target all -j 12
cmake --build ./build/Debug --config Debug --target all -j 12