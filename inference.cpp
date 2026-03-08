// // Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
// //CUDA + TensorRT
//
#include "inference.h"
#include <regex>
#include <algorithm>
#include <filesystem>

#define benchmark
#define min(a,b) (((a) < (b)) ? (a) : (b))

YOLO_V8::YOLO_V8() {
}

YOLO_V8::~YOLO_V8() {
   delete session;
}

#ifdef USE_CUDA
namespace Ort
{
   template<>
   struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif

template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
   int channels = iImg.channels();
   int imgHeight = iImg.rows;
   int imgWidth = iImg.cols;

   for (int c = 0; c < channels; c++)
   {
       for (int h = 0; h < imgHeight; h++)
       {
           for (int w = 0; w < imgWidth; w++)
           {
               iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                   (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
           }
       }
   }
   return RET_OK;
}

char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
   if (iImg.channels() == 3)
   {
       oImg = iImg.clone();
       cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
   }
   else
   {
       cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
   }

   switch (modelType)
   {
   case YOLO_DETECT_V8:
   case YOLO_POSE:
   case YOLO_DETECT_V8_HALF:
   case YOLO_POSE_V8_HALF:
   case YOLO_DETECT_SEG:
   {
       if (iImg.cols >= iImg.rows)
       {
           resizeScales = iImg.cols / (float)iImgSize.at(0);
           cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
       }
       else
       {
           resizeScales = iImg.rows / (float)iImgSize.at(0);
           cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
       }
       cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
       oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
       oImg = tempImg;
       break;
   }
   case YOLO_CLS:
   {
       int h = iImg.rows;
       int w = iImg.cols;
       int m = min(h, w);
       int top = (h - m) / 2;
       int left = (w - m) / 2;
       cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
       break;
   }
   }
   return RET_OK;
}

char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
   char* Ret = RET_OK;
   std::regex pattern("[\u4e00-\u9fa5]");
   bool result = std::regex_search(iParams.modelPath, pattern);
   if (result)
   {
       Ret = "[YOLO_V8]:Your model path is error.Change your model path without chinese characters.";
       std::cout << Ret << std::endl;
       return Ret;
   }
   try
   {
       rectConfidenceThreshold = iParams.rectConfidenceThreshold;
       iouThreshold = iParams.iouThreshold;
       imgSize = iParams.imgSize;
       modelType = iParams.modelType;
       cudaEnable = iParams.cudaEnable;
       useTensorRT = iParams.useTensorRT;
       useFP16 = iParams.useFP16;

       // Initialize class names
       if (modelType == YOLO_DETECT_V8 || modelType == YOLO_DETECT_V8_HALF || modelType == YOLO_DETECT_SEG) {
           classes = {
               "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
               "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
               "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
               "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
               "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
               "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
               "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
           };
           std::cout << "[YOLO_V8]: Loaded " << classes.size() << " class names" << std::endl;
       }

       env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
       Ort::SessionOptions sessionOption;


       // Configure execution providers
       if (iParams.cudaEnable)
       {
           std::cout << "[INFO] Attempting to enable GPU acceleration..." << std::endl;

           bool gpuConfigured = false;

           // Try TensorRT first if requested
           if (iParams.useTensorRT)
           {
               try {
                   std::cout << "[INFO] Configuring TensorRT provider..." << std::endl;
                   OrtTensorRTProviderOptions trt_options{};
                   trt_options.device_id = 0;
                   trt_options.trt_max_workspace_size = 2ULL << 30; // 2GB
                   trt_options.trt_fp16_enable = iParams.useFP16 ? 1 : 0;
                   trt_options.trt_engine_cache_enable = 1;
                   trt_options.trt_engine_cache_path = iParams.trtCachePath.c_str();

                   // Fix dynamic shapes - allow TensorRT to handle variable sizes
                   trt_options.trt_max_partition_iterations = 1000;
                   trt_options.trt_min_subgraph_size = 1;
                   trt_options.trt_dla_enable = 0;

                   // Create cache directory if it doesn't exist
                   std::filesystem::create_directories(iParams.trtCachePath);

                   sessionOption.AppendExecutionProvider_TensorRT(trt_options);
                   std::cout << " TensorRT provider configured successfully!" << std::endl;
                   std::cout << " FP16: " << (iParams.useFP16 ? "ENABLED" : "DISABLED") << std::endl;
                   std::cout << " Cache path: " << iParams.trtCachePath << std::endl;
                   gpuConfigured = true;
               }
               catch (const std::exception& e) {
                   std::cout << " TensorRT provider failed: " << e.what() << std::endl;
                   std::cout << "  Falling back to CUDA..." << std::endl;
               }
           }

           // Configure CUDA (always, as fallback or primary)
           try {
               std::cout << "[INFO] Configuring CUDA provider..." << std::endl;
               OrtCUDAProviderOptions cudaOption;
               cudaOption.device_id = 0;
               cudaOption.arena_extend_strategy = 0;
               cudaOption.gpu_mem_limit = SIZE_MAX;
               cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
               cudaOption.do_copy_in_default_stream = 1;

               sessionOption.AppendExecutionProvider_CUDA(cudaOption);
               std::cout << " CUDA provider configured successfully!" << std::endl;
               std::cout << " Device ID: 0" << std::endl;
               gpuConfigured = true;
           }
           catch (const std::exception& e) {
               std::cout << "✗ CUDA provider failed: " << e.what() << std::endl;
               std::cout << "✗ GPU ACCELERATION FAILED!" << std::endl;
               std::cout << "  Will fall back to CPU (this will be SLOW)" << std::endl;
               cudaEnable = false;
           }

           if (gpuConfigured) {
               std::cout << "\n GPU ACCELERATION ACTIVE" << std::endl;
           }
       }
       else
       {
           std::cout << "[INFO] Running on CPU (GPU disabled)" << std::endl;
       }



       sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
       sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
       sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
       int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
       wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
       MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
       wide_cstr[ModelPathSize] = L'\0';
       const wchar_t* modelPath = wide_cstr;
#else
       const char* modelPath = iParams.modelPath.c_str();
#endif

       std::cout << "[INFO] Loading ONNX model..." << std::endl;
       session = new Ort::Session(env, modelPath, sessionOption);
       std::cout << " Model loaded successfully!" << std::endl;

#ifdef _WIN32
       delete[] wide_cstr;
#endif

       Ort::AllocatorWithDefaultOptions allocator;
       size_t inputNodesNum = session->GetInputCount();
       for (size_t i = 0; i < inputNodesNum; i++)
       {
           Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
           char* temp_buf = new char[50];
           strcpy(temp_buf, input_node_name.get());
           inputNodeNames.push_back(temp_buf);
       }
       size_t OutputNodesNum = session->GetOutputCount();
       for (size_t i = 0; i < OutputNodesNum; i++)
       {
           Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
           char* temp_buf = new char[10];
           strcpy(temp_buf, output_node_name.get());
           outputNodeNames.push_back(temp_buf);
       }
       options = Ort::RunOptions{ nullptr };

       std::cout << "\n[INFO] Warming up session..." << std::endl;
       WarmUpSession();
       std::cout << "Session ready!\n" << std::endl;

       return RET_OK;
   }
   catch (const std::exception& e)
   {
       const char* str1 = "[YOLO_V8]:";
       const char* str2 = e.what();
       std::string result = std::string(str1) + std::string(str2);
       char* merged = new char[result.length() + 1];
       std::strcpy(merged, result.c_str());
       std::cout << merged << std::endl;
       delete[] merged;
       return "[YOLO_V8]:Create session failed.";
   }
}

char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
   clock_t starttime_1 = clock();
#endif

   char* Ret = RET_OK;
   cv::Mat processedImg;
   PreProcess(iImg, imgSize, processedImg);

   if (modelType < 4 || modelType == YOLO_DETECT_SEG)
   {
       float* blob = new float[processedImg.total() * 3];
       BlobFromImage(processedImg, blob);
       std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
       TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
   }
   else
   {
#ifdef USE_CUDA
       half* blob = new half[processedImg.total() * 3];
       BlobFromImage(processedImg, blob);
       std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
       TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
   }

   return Ret;
}

template<typename N>

char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
   std::vector<DL_RESULT>& oResult) {
   Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
       Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
       inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
   clock_t starttime_2 = clock();
#endif
   auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
       outputNodeNames.size());
#ifdef benchmark
   clock_t starttime_3 = clock();
#endif

   Ort::TypeInfo typeInfo = outputTensor[0].GetTypeInfo();
   auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
   std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
   auto output = outputTensor[0].GetTensorMutableData<typename std::remove_pointer<N>::type>();
   delete[] blob;

   switch (modelType)
   {
   case YOLO_DETECT_V8:
   case YOLO_DETECT_V8_HALF:
   case YOLO_DETECT_SEG:
   {
       int signalResultNum = outputNodeDims[1];
       int strideNum = outputNodeDims[2];
       std::vector<int> class_ids;
       std::vector<float> confidences;
       std::vector<cv::Rect> boxes;
       std::vector<std::vector<float>> mask_coeffs;

       cv::Mat rawData;
       if (modelType == YOLO_DETECT_V8 || modelType == YOLO_DETECT_SEG)
       {
           rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
       }
       else
       {
           rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
           rawData.convertTo(rawData, CV_32F);
       }
       rawData = rawData.t();

       float* data = (float*)rawData.data;

       for (int i = 0; i < strideNum; ++i){
           float* classesScores = data + 4;
           cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
           cv::Point class_id;
           double maxClassScore;
           cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

           if (maxClassScore > rectConfidenceThreshold){
               confidences.push_back(maxClassScore);
               class_ids.push_back(class_id.x);

               float x = data[0];
               float y = data[1];
               float w = data[2];
               float h = data[3];

               int left = int((x - 0.5 * w) * resizeScales);
               int top = int((y - 0.5 * h) * resizeScales);
               int width = int(w * resizeScales);
               int height = int(h * resizeScales);

               boxes.push_back(cv::Rect(left, top, width, height));

               if (modelType == YOLO_DETECT_SEG && signalResultNum >= 116){
                   std::vector<float> coeffs;
                   for (int c = 0; c < 32; c++) {
                       coeffs.push_back(data[4 + this->classes.size() + c]);
                   }
                   mask_coeffs.push_back(coeffs);
               }
           }
           data += signalResultNum;
       }

       std::vector<int> nmsResult;
       cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);

       cv::Mat mask_protos;
       if (modelType == YOLO_DETECT_SEG && outputTensor.size() > 1)
       {
           auto mask_info = outputTensor[1].GetTensorTypeAndShapeInfo();
           auto mask_dims = mask_info.GetShape();
           int proto_h = static_cast<int>(mask_dims[2]);
           int proto_w = static_cast<int>(mask_dims[3]);
           int num_protos = static_cast<int>(mask_dims[1]);

           auto mask_data = outputTensor[1].GetTensorMutableData<float>();
           mask_protos = cv::Mat(num_protos, proto_h * proto_w, CV_32F, mask_data);
       }

       for (int i = 0; i < nmsResult.size(); ++i){
           int idx = nmsResult[i];
           DL_RESULT result;
           result.classId = class_ids[idx];
           result.confidence = confidences[idx];
           result.box = boxes[idx];

           if (modelType == YOLO_DETECT_SEG && !mask_protos.empty() && idx < mask_coeffs.size())
           {
               int proto_h = 160;
               int proto_w = 160;
               int num_protos = 32;

               cv::Mat protos = mask_protos.reshape(1, num_protos).t();
               cv::Mat coeffs(1, num_protos, CV_32F, mask_coeffs[idx].data());
               cv::Mat mask_flat = coeffs * protos.t();
               cv::Mat mask = mask_flat.reshape(1, proto_h);

               cv::exp(-mask, mask);
               mask = 1.0 / (1.0 + mask);

               cv::Rect box = result.box & cv::Rect(0, 0, iImg.cols, iImg.rows);

               float scale_x = (float)proto_w / imgSize[1];
               float scale_y = (float)proto_h / imgSize[0];

               cv::Rect scaled_box(
                   int(box.x * scale_x),
                   int(box.y * scale_y),
                   int(box.width * scale_x),
                   int(box.height * scale_y)
               );

               scaled_box &= cv::Rect(0, 0, proto_w, proto_h);
               cv::Mat mask_crop = mask(scaled_box);
               cv::Mat mask_resized;
               cv::resize(mask_crop, mask_resized, cv::Size(box.width, box.height));

               cv::Mat binMask;
               cv::threshold(mask_resized, binMask, 0.5, 255.0, cv::THRESH_BINARY);
               binMask.convertTo(binMask, CV_8U);

               result.mask = cv::Mat::zeros(iImg.size(), CV_8U);
               binMask.copyTo(result.mask(box));
           }

           oResult.push_back(result);
       }

#ifdef benchmark
       clock_t starttime_4 = clock();
       double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
       double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
       double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;

       std::string backend = "CPU";
       if (cudaEnable) {
           backend = useTensorRT ? "TensorRT+CUDA" : "CUDA";
       }

       std::cout << "[" << backend << "] "
                 << pre_process_time << "ms pre | "
                 << process_time << "ms inference | "
                 << post_process_time << "ms post" << std::endl;
#endif

       break;
   }
   case YOLO_CLS:
   case YOLO_CLS_HALF:
   {
       cv::Mat rawData;
       if (modelType == YOLO_CLS) {
           rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
       } else {
           rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
           rawData.convertTo(rawData, CV_32F);
       }
       float *data = (float *) rawData.data;

       DL_RESULT result;
       for (int i = 0; i < this->classes.size(); i++)
       {
           result.classId = i;
           result.confidence = data[i];
           oResult.push_back(result);
       }
       break;
   }
   default:
       std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
   }
   return RET_OK;
}

char* YOLO_V8::WarmUpSession() {
   clock_t starttime_1 = clock();
   cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
   cv::Mat processedImg;
   PreProcess(iImg, imgSize, processedImg);

   if (modelType < 4 || modelType == YOLO_DETECT_SEG)
   {
       float* blob = new float[iImg.total() * 3];
       BlobFromImage(processedImg, blob);
       std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
       Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
           Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
           YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
       auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
           outputNodeNames.size());
       delete[] blob;
       clock_t starttime_4 = clock();
       double warm_up_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;

       std::string backend = "CPU";
       if (cudaEnable) {
           backend = useTensorRT ? "TensorRT+CUDA" : "CUDA";
       }
       std::cout << "[" << backend << "] Warm-up completed in " << warm_up_time << " ms" << std::endl;
   }
   else
   {
#ifdef USE_CUDA
       half* blob = new half[iImg.total() * 3];
       BlobFromImage(processedImg, blob);
       std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
       Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
       auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
       delete[] blob;
       clock_t starttime_4 = clock();
       double warm_up_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
       std::string backend = useTensorRT ? "TensorRT+CUDA" : "CUDA";
       std::cout << "[" << backend << "] Warm-up completed in " << warm_up_time << " ms" << std::endl;
#endif
   }
   return RET_OK;
}

