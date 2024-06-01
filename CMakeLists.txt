cmake_minimum_required(VERSION 3.0)
project(MyProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find packages
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable
add_executable(my_app inference.cpp)

# Link libraries
target_link_libraries(my_app ${OpenCV_LIBS})
target_link_libraries(my_app nlohmann_json::nlohmann_json)
target_link_libraries(my_app pthread)
