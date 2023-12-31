# Crack-Detection-Team-Project

**Team Members:** Luying Ruan, Jim Stiegler, Hua Tong

**Mission**: The original mission of our project was to enable the Architecture, Engineering, and Construction (AEC) industry to safely and efficiently inspect structures to detect and characterize potential anomalies that require attention.

To fulfill this mission, we originally aimed to automate a drone to detect cracks on a building and/or roof. However, due to the complexity of such an endeavor and limited time/resources, we scoped our project to enhancing an existing crack detection capability by further training a pre-built model. Our team consisted of BU ECE and SE students with very limited familiarity in machine learning and ML models. Therefore, we took this opportunity to learn from existing projects and try to conduct more of an analytic assessment and comparision of various crack detection capabilities.

**Project Deliverables**: See our Sprint 1 and 2 pptx for overviews of our project, goals, user stories, milestones, etc. We also provide the poster (poster final.pdf) we created for a student project forum held on 8 December 2023 in the Photonics building at Boston University.

**Capability Users:**
Architects
Construction Workers
Structural Engineers
Customer (Homeowner, Commercial, Government)
Regulatory Entities (Local, State, Federal)

**Minimum Viable Product:** Our MVP was to take an existing model and retrain it with a more varied dataset to further refine the model. This dataset would then produce more accurate detection results.

**Primary Project Analyzed and Trained: Yolov5Mode** - In this project, the research team developed a model to detect a mask on a person. We piggybacked off this project, but modified and refined the model to detect cracks in images. We trained a model using a new dataset with different labels. We found a varied dataset of images with various sizes, shapes, and colors of cracks on different surfaces. 

**Secondary Project Analyzed:** While not the primary focus of the project, we did compare our enhanced trained Yolo5 Model with a model trained using the DeepCrack project, which can be found here: https://github.com/qinnzou/DeepCrack 

For DeepCrack, you can use the crack_detector.py and crack_detector_v2.py files to run the DeepCrack model from (found in the DeepCrack project here: [https://github.com/qinnzou/DeepCrack](https://github.com/qinnzou/DeepCrack#pretrained-models))

**Setup instructions:**
1. Download and install the latest 7-Zip: https://www.7-zip.org/
2. Download the Yolov5Mode.7z.xxx files, the dataset.zip.xxx, and Dev-20231206T203908Z-001.zip.
3. Right click on Yolov5Mode.7z.001 and dataset.zip.001 and click "Combine" to rebuild the Zip file. Extract the respective Yolov5Mode and dataset zip files.
4. Load files into your preferred environment (we used PyCharm)
5. Ensure you have the latest torch, Python, etc.
6. Follow additional setup instructions via the README.md file found in the Yolov5Mode folder.
7. For DeepCrack, follow the instructions provided on the DeepCrack Github page.

**Datasets:** As noted, we used the images provided in the dataset.zip file. We also used images provided by the DeepCrack project, found here: https://github.com/qinnzou/DeepCrack#download.

**License:** The Yolov5 project uses a GNU GENERAL PUBLIC LICENSE. See the LICENSE file for more info. 

**Citation:**
@article{zou2018deepcrack,
  title={Deepcrack: Learning Hierarchical Convolutional Features for Crack Detection},
  author={Zou, Qin and Zhang, Zheng and Li, Qingquan and Qi, Xianbiao and Wang, Qian and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={3},
  pages={1498--1512},
  year={2019},
}

@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}
