1. Make sure the program compiles without error, i.e. the commands in `compile.sh` should return without error.
2. Build the GPU index. Then compare the performance with the CPU-built version. The recall shouldn't differ much.
  For this to pass, run the following commands and compare the output recall of GPU and CPU.
  ```bash
  ./build/Release/src/benchs/bench-build-gpu --datacard sift --k 10 --M 16 --efc 100 --profile
  ./build/Release/src/benchs/bench-index-search --datacard sift --k 10 --hardware GPU --M 16 --efc 100 --efs 100
  ./build/Release/src/benchs/bench-index-search --datacard sift --k 10 --hardware CPU --M 16 --efc 100 --efs 100


  ./build/Release/src/benchs/bench-build-gpu --datacard video-dedup --k 10 --M 16 --efc 100 --profile
  ./build/Release/src/benchs/bench-index-search --datacard video-dedup --k 10 --hardware GPU --M 16 --efc 100 --efs 100
  ./build/Release/src/benchs/bench-index-search --datacard video-dedup --k 10 --hardware CPU --M 16 --efc 100 --efs 100
  ```