"
-------------------------English Version-------------------------------
# Overview and Methodology  


Initial implementation and testing were performed on a desktop platform equipped with an NVIDIA GeForce RTX 3050 GPU and processor.

## 2. Solution Architecture  


The chosen model was YOLOv8 small due to platform constraints. It occupies little space in ONNX format and does not consume a lot of memory.  
The main technologies used are ONNX Runtime and TensorRT. TensorRT has the ability to transform an FP32 format into a lower precision format such as FP16 or INT8. Following this reduction, lower latency is achieved.

## 3. Architecture Block Diagram  


* Data source: a static image
* Inference: detects objects, draws bounding boxes, confidence scores, and applies masks
* Post-processing: applies a confidence factor, removes overlaps, and redraws masks
* Display module: drawing bounding boxes and masks on the image
* Optimization: CUDA and TensorRT, FP16 activation, and warm-up phase

## 4. Implemented Algorithms  


The segmentation algorithm uses mask coefficients and network-generated prototypes to obtain the mask for each detected object. Finally, this mask is binarized using a preset threshold.

## 5. Experimental Results  


* The first variant was running the algorithm on the CPU to validate a project baseline and ensure correct object detection and segmentation in an image.
* The second variant was using CUDA to run the algorithm on the GPU. A significant performance increase in speed and inference could be observed.
* The third and final variant was using CUDA with acceleration via the TensorRT engine. Following this test, the GPU was accelerated, delivering maximum performance.
* <img width="609" height="109" alt="image" src="https://github.com/user-attachments/assets/7e6a0a46-8fd0-4eb1-8633-518251831fea" />
-------------------------Varianta în română---------------------------- 


#1. Prezentare Generală și Metodologie  


Implementarea și testarea inițiale s-au realizat pe o platformă desktop echipată cu GPU NVIDIA GEFORCE RTX 3050 și procesor. 


#2. Arhitectura soluției  


   Modelul ales a fost YOLOv8 small din cauza constrângerilor platformei. Ocupă puțin spațiu în format ONNX și nu consumă multă memorie 
Tehnologiile principale folosite sunt ONNX Runtime și TensorRT. TensorRT are abilitatea de a transforma un format FP32 într-un format cu precizie mai mică precum FP16 sau INT8. În urma acestei reduceri se obține o latență mai scăzută. 

#3. Schema bloc a arhitecturii  


  1. Sursa de date: o imagine statică
  2. Inferența: detectează obiectele, deseneaza bounding box-urile, scoruri de încredere și aplică măști
  3. Postprocesarea: se aplică un factor de încredere, elimină suprapunerile și redesenează măștile
  4. Modulul de afișare: desenarea bounding box-urilor și a măștilor pe imagine
  5. Optimizarea: CUDA și TensorRT, activarea FP16 și etapa de warm-up

#4. Algoritmii implementați   


  Algoritmul de segmentare folosește coeficienți de mască și prototipuri generate de rețea 
pentru a obține masca pentru fiecare obiect detectat. În final, această mască este binarizată 
folosind un prag prestabilit. 

#5. Rezultate experimentale  


  1. Prima variantă a fost rularea algoritmului pe CPU pentru a valida o bază a proiectului 
și a asigura detectarea și segmentarea corecte ale obiectelor într-o imagine.  
  2. A doua variantă a fost folosirea CUDA pentru a putea rula algoritmul pe GPU. S-a putut 
observa o creștere semnificativă de performanță în ceea ce privește viteza și inferența.  
  3. A treia și ultima variantă a fost folosirea CUDA cu accelerare folosind engine-ul 
TensorRT. În urma acestei testări, GPU-ul a fost accelerat, oferind performanțe maxime.

<img width="609" height="109" alt="image" src="https://github.com/user-attachments/assets/7e6a0a46-8fd0-4eb1-8633-518251831fea" />





