# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jupyter/Notebooks/Rodrigo/hanabi-learning-environment

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jupyter/Notebooks/Rodrigo/hanabi-learning-environment

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jupyter/Notebooks/Rodrigo/hanabi-learning-environment/CMakeFiles /home/jupyter/Notebooks/Rodrigo/hanabi-learning-environment/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jupyter/Notebooks/Rodrigo/hanabi-learning-environment/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named game_example

# Build rule for target.
game_example: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 game_example
.PHONY : game_example

# fast build rule for target.
game_example/fast:
	$(MAKE) -f CMakeFiles/game_example.dir/build.make CMakeFiles/game_example.dir/build
.PHONY : game_example/fast

#=============================================================================
# Target rules for targets named pyhanabi

# Build rule for target.
pyhanabi: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 pyhanabi
.PHONY : pyhanabi

# fast build rule for target.
pyhanabi/fast:
	$(MAKE) -f CMakeFiles/pyhanabi.dir/build.make CMakeFiles/pyhanabi.dir/build
.PHONY : pyhanabi/fast

#=============================================================================
# Target rules for targets named hanabi

# Build rule for target.
hanabi: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 hanabi
.PHONY : hanabi

# fast build rule for target.
hanabi/fast:
	$(MAKE) -f hanabi_lib/CMakeFiles/hanabi.dir/build.make hanabi_lib/CMakeFiles/hanabi.dir/build
.PHONY : hanabi/fast

game_example.o: game_example.cc.o

.PHONY : game_example.o

# target to build an object file
game_example.cc.o:
	$(MAKE) -f CMakeFiles/game_example.dir/build.make CMakeFiles/game_example.dir/game_example.cc.o
.PHONY : game_example.cc.o

game_example.i: game_example.cc.i

.PHONY : game_example.i

# target to preprocess a source file
game_example.cc.i:
	$(MAKE) -f CMakeFiles/game_example.dir/build.make CMakeFiles/game_example.dir/game_example.cc.i
.PHONY : game_example.cc.i

game_example.s: game_example.cc.s

.PHONY : game_example.s

# target to generate assembly for a file
game_example.cc.s:
	$(MAKE) -f CMakeFiles/game_example.dir/build.make CMakeFiles/game_example.dir/game_example.cc.s
.PHONY : game_example.cc.s

pyhanabi.o: pyhanabi.cc.o

.PHONY : pyhanabi.o

# target to build an object file
pyhanabi.cc.o:
	$(MAKE) -f CMakeFiles/pyhanabi.dir/build.make CMakeFiles/pyhanabi.dir/pyhanabi.cc.o
.PHONY : pyhanabi.cc.o

pyhanabi.i: pyhanabi.cc.i

.PHONY : pyhanabi.i

# target to preprocess a source file
pyhanabi.cc.i:
	$(MAKE) -f CMakeFiles/pyhanabi.dir/build.make CMakeFiles/pyhanabi.dir/pyhanabi.cc.i
.PHONY : pyhanabi.cc.i

pyhanabi.s: pyhanabi.cc.s

.PHONY : pyhanabi.s

# target to generate assembly for a file
pyhanabi.cc.s:
	$(MAKE) -f CMakeFiles/pyhanabi.dir/build.make CMakeFiles/pyhanabi.dir/pyhanabi.cc.s
.PHONY : pyhanabi.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... game_example"
	@echo "... rebuild_cache"
	@echo "... pyhanabi"
	@echo "... hanabi"
	@echo "... game_example.o"
	@echo "... game_example.i"
	@echo "... game_example.s"
	@echo "... pyhanabi.o"
	@echo "... pyhanabi.i"
	@echo "... pyhanabi.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

