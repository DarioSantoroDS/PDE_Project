#!/bin/bash

module purge
module load gcc-glibc
module load dealii

# forward all arguments to cmake
exec cmake "$@"
