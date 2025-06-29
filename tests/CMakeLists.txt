
# add_executable(mpm_ball_free_fall_test mpm_ball_free_fall_test.cu ${CMAKE_SOURCE_DIR}/src/cuda_util.cu ${CMAKE_SOURCE_DIR}/src/mpm_engine.cu)
# target_link_libraries(mpm_ball_free_fall_test partio)

set(MPM_SIMULATOR_SOURCE_FILE ${CMAKE_SOURCE_DIR}/src/cuda_util.cu)

# Build all test cases
set(MPM_TEST_CASES "dragon;fluid_dam_break;pumpkin;oreo;colliding_sphere;rotating_rod;sand_armadillo;droplet;fluid_with_two_longs;bullet;castle_crasher;twisting_bar;chimney_ball;fire_hydrant")
foreach(TEST_CASE_NAME ${MPM_TEST_CASES})
    message("Adding test case --- ${TEST_CASE_NAME}")
    set(TEST_TARGET_NAME mpm_test_${TEST_CASE_NAME})
    add_executable(${TEST_TARGET_NAME} ${TEST_CASE_NAME}/test_${TEST_CASE_NAME}.cu ${MPM_SIMULATOR_SOURCE_FILE})
    target_link_libraries(${TEST_TARGET_NAME} partio)
endforeach()

