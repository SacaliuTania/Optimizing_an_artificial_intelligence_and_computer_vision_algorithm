cmake_minimum_required(VERSION 3.18)
project(yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/facultate/An3/An3_sem1/SSC/proiect_ssc/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

#onnx
set(ONNXRUNTIME_DIR "D:/facultate/An3/An3_sem1/SSC/proiect_ssc/onnxruntime_gpu")

#opencv
#set(OpenCV_DIR "D:/facultate/An3/An3_sem1/SSC/proiect_ssc/opencv/build")

#cuda
#set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")


#cudnn
set(CUDNN_DIR "C:/Program Files/NVIDIA/CUDNN/v9.16")
include_directories(${CUDNN_DIR}/include)
link_directories(${CUDNN_DIR}/lib/12.9/x64)

#tensor
set(TENSORRT_DIR "D:/facultate/An3/An3_sem1/SSC/proiect_ssc/TensorRT")
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

find_package(OpenCV REQUIRED)


include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

add_executable(yolo
        main.cpp
        inference.cpp
        det_boundingBox.cu
        segm.cu
        gpu_draw.h
        gpu_draw.cu
        inference_gpu.h
        kernel.cu
        inference_gpu.cpp
        kernels.h)

target_link_libraries(yolo
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
        ${OpenCV_LIBS}
        "${TENSORRT_DIR}/lib/nvinfer_10.lib"
        "${TENSORRT_DIR}/lib/nvonnxparser_10.lib"

        CUDA::cudart
        CUDA::cublas
        cudnn
)

if(WIN32)

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )



    #add_custom_command(TARGET yolo POST_BUILD
    #        COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #        "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120d.dll"
    #        "$<TARGET_FILE_DIR:yolo>"
    #)

    set(OPENCV_DLL "D:/facultate/An3/An3_sem1/SSC/proiect_ssc/opencv/build/x64/vc16/bin/opencv_world4120d.dll")
    if(EXISTS ${OPENCV_DLL})
        add_custom_command(TARGET yolo POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DLL}" "$<TARGET_FILE_DIR:yolo>"
        )
    else()
        message(WARNING "OpenCV DLL not found: ${OPENCV_DLL}")
    endif()


    #onnxruntime_gpu
    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_tensorrt.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )


    #cudnn
    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDNN_DIR}/bin/12.9/cudnn64_9.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDNN_DIR}/bin/12.9/cudnn_ops64_9.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDNN_DIR}/bin/12.9/cudnn_cnn64_9.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

    add_custom_command(TARGET yolo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDNN_DIR}/bin/12.9/cudnn_adv64_9.dll"
            "$<TARGET_FILE_DIR:yolo>"
    )

endif()