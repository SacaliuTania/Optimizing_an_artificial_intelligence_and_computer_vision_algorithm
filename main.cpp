#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include "inference.h"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <onnxruntime_c_api.h>
#include "gpu_draw.h"

void performanceCheck(YOLO_V8 *detector) {
   cv::Mat test_img = cv::Mat::zeros(640, 640, CV_8UC3);
   std::vector<DL_RESULT> results;

   // Warm up
   detector->RunSession(test_img, results);
   results.clear();


   std::cout << "10 inferente" << std::endl;

   // masoara 10 inferente
   auto start = std::chrono::high_resolution_clock::now();
   for (int i = 0; i < 10; i++) {
       detector->RunSession(test_img, results);
       results.clear();
   }
   auto end = std::chrono::high_resolution_clock::now();

   double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
   double avg_ms = total_ms / 10.0;
   double fps = 1000.0 / avg_ms;


   std::cout << "Average time per image: " << std::fixed << std::setprecision(2) << avg_ms << " ms" << std::endl;
   std::cout << "Approximate FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;

   std::cout << "\nPerformance Analysis:" << std::endl;
   if (avg_ms < 30) {
       std::cout << "GPU optim" << std::endl;
   } else if (avg_ms < 100) {
       std::cout << "Accelerare GPU activata" << std::endl;
   } else if (avg_ms < 300) {
       std::cout <<"GPU ul nu este utilizat 100 %" << std::endl;
   } else {
       std::cout << "ruleaza pe CPU " << std::endl;
   }

}

using namespace cv;
using namespace std;

void Detector(YOLO_V8*& p) {
   std::filesystem::path imgs_path = "D:/facultate/An3/An3_sem1/SSC/proiect_ssc/coco8/coco8/images/val";

   if (!std::filesystem::exists(imgs_path)) {
       std::cerr << "Image path does not exist: " << imgs_path << std::endl;
       return;
   }

   for (auto& i : std::filesystem::directory_iterator(imgs_path)) {
       if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
       {
           std::string img_path = i.path().string();
           cv::Mat img = cv::imread(img_path);

           if (img.empty()) {
               std::cerr << "Could not load: " << img_path << std::endl;
               continue;
           }

           std::cout << "\nProcesare: " << i.path().filename()<< std::endl;
           std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

           std::vector<DL_RESULT> res;
           p->RunSession(img, res); //transfera datele din CPU RAM pe GPU VRAM

           std::cout << "Detections: " << res.size() << std::endl;

           // masca
           for (auto& re : res)
           {
               cv::RNG rng(cv::getTickCount());
               cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

               int h = (img.rows < re.mask.rows) ? img.rows : re.mask.rows;
               int w = (img.cols < re.mask.cols) ? img.cols : re.mask.cols;

               for (int y = 0; y < h; ++y)
               {
                   for (int x = 0; x < w; ++x)
                   {
                       if (re.mask.at<uchar>(y, x) > 0)
                       {
                           img.at<cv::Vec3b>(y, x) =
                               img.at<cv::Vec3b>(y, x) * 0.5 +
                               cv::Vec3b(color[0], color[1], color[2]) * 0.5;
                       }
                   }
               }
           }

           //  bounding boxes
           for (auto& re : res)
           {
               cv::RNG rng(cv::getTickCount());
               cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

               cv::rectangle(img, re.box, color, 3);

               float confidence = floor(100 * re.confidence) / 100;
               std::ostringstream ss;
               ss << std::fixed << std::setprecision(2) << confidence;
               std::string label = p->classes[re.classId] + " " + ss.str();

               int baseLine = 0;
               cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseLine);

               cv::rectangle(
                   img,
                   cv::Point(re.box.x, re.box.y - textSize.height - 8),
                   cv::Point(re.box.x + textSize.width, re.box.y),
                   color,
                   cv::FILLED
               );

               cv::putText(
                   img,
                   label,
                   cv::Point(re.box.x, re.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX,
                   0.75,
                   cv::Scalar(0, 0, 0),
                   2
               );
           }

           std::cout << "Press any key to continue (ESC to exit)" << std::endl;
           cv::imshow("Result of Detection", img);
           int key = cv::waitKey(0);
           if (key == 27) {
               cv::destroyAllWindows();
               return;
           }
       }
   }
   cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {

   std::cout << "Verificare GPU" << std::endl;
   std::cout << "ONNX Runtime Version: " << OrtGetApiBase()->GetVersionString() << std::endl;

   try {
       std::string model_path;
       if (argc < 2) {
           model_path = R"(D:\facultate\An3\An3_sem1\SSC\proiect_ssc\yolov8s-seg.onnx)";
       }
       else {
           model_path = argv[1];
       }

       if (!std::filesystem::exists(model_path)) {
           std::cerr << "Model not found at: " << model_path << std::endl;
           return 1;
       }

       std::cout << "Model found: " << model_path << std::endl;

       // Initialize YOLO detector
       YOLO_V8* yolo_detector = new YOLO_V8();
       DL_INIT_PARAM params;
       params.modelPath = model_path;
       params.modelType = YOLO_DETECT_SEG;
       params.imgSize = {640, 640};
       params.rectConfidenceThreshold = 0.25f;
       params.iouThreshold = 0.3f;
       params.intraOpNumThreads = 1;
       params.logSeverityLevel = 3;


       //CUDA + TensorRT
       params.cudaEnable = true;
       params.useTensorRT = true;
       params.useFP16 = true;

       //doar CUDA
       //params.cudaEnable = true;
       //params.useTensorRT = false;
       //params.useFP16 = false;

       //CPU
       // params.cudaEnable = false;
       // params.useTensorRT = false;
       // params.useFP16 = false;

       char* result = yolo_detector->CreateSession(params);
       if (result != RET_OK) {
           std::cerr << "\n✗ Failed to create YOLO session: " << result << std::endl;
           delete yolo_detector;
           return 1;
       }

       performanceCheck(yolo_detector);

       std::cout << "\nStarting image detection...\n" << std::endl;
       Detector(yolo_detector);

       std::cout << "\nDetection completed!" << std::endl;

       delete yolo_detector;
   }
   catch (const Ort::Exception& e) {
       std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
       return 1;
   }
   catch (const std::exception& e) {
       std::cerr << "Error: " << e.what() << std::endl;
       return 1;
   }

   return 0;
}

