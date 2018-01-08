cmake_minimum_required(VERSION 3.9)
project(ml_test)

set(CMAKE_CXX_STANDARD 11)

#add_executable(ml_test  decisionTree.cpp)
add_executable(ml_test  LDA.cpp)
find_package(OpenCV REQUIRED)                       # 搜索指定的外部依赖库头文件和库文件
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )   # 链接到OpenCV库 ${jsoncpp_LIBS}