#!/bin/bash

# 1. Initialize Modules
if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
elif [ -f /etc/profile.d/lmod.sh ]; then
   source /etc/profile.d/lmod.sh
else
   source ~/.bashrc
fi

# 2. Load the required modules
module load gcc-glibc dealii

# 3. CRITICAL: Capture the library path we just generated
# We must capture it here because VS Code might try to reset it later.
libs="$LD_LIBRARY_PATH"

# 4. Fix Signal handling so the "Pause" button works in VS Code
# trap - SIGINT SIGTERM

# 5. Launch GDB
# We use '-ex "set env ..."' to force the environment variable INSIDE GDB.
# This prevents VS Code from overwriting it with its own settings.
exec /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/gdb \
    -ex "set env LD_LIBRARY_PATH=$libs" \
    "$@"