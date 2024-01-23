include_directories(${CMAKE_SOURCE_DIR}/)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE bert ggml)

add_executable(simple simple.cpp)
target_link_libraries(simple PRIVATE bert ggml)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE bert ggml)

add_executable(test_batch_encode test_batch_encode.cpp)
target_link_libraries(test_batch_encode PRIVATE bert ggml)
