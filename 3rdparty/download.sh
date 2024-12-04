#!/bin/sh
set -e

wget https://github.com/gabime/spdlog/archive/refs/tags/v1.8.5.zip -O spdlog-1.8.5.zip
unzip spdlog-1.8.5.zip
mv spdlog-1.8.5 spdlog

wget https://github.com/rapidsai/rmm/archive/refs/tags/v22.04.00.zip -O rmm-22.04.00.zip
unzip rmm-22.04.00.zip
mv rmm-22.04.00 rmm

wget https://github.com/rapidsai/raft/archive/refs/tags/v23.04.00.zip -O raft-23.04.00.zip
unzip raft-23.04.00.zip
mv raft-23.04.00 raft
rm rmm-22.04.00.zip spdlog-1.8.5.zip raft-23.04.00.zip