# Crack-Detection-Team-Project

**Mission**: The original mission of our project was to enable the Architecture, Engineering, and Construction (AEC) industry to safely and efficiently inspect structures to detect and characterize potential anomalies that require attention.

To fulfill this mission we aimed to automate a drone to detect cracks on a building and/or roof. However, due to the complexity of such an endeavor, we scoped our project to enhancing an existing crack detection capability by further training a pre-built model. Our team consisted of ECE and SE students that did not have much familiarity in machine learning and ML models. Therefore, we took this opportunity to learn from existing projects and try to conduct more of an analytic assessment and comparision of various crack detection capabilities.

**Capability Users:**
Architects
Construction Workers
Structural Engineers
Customer (Homeowner, Commercial, Government)
Regulatory Entities (Local, State, Federal)

**Minimum Viable Product:** Our MVP was to take an existing model and retrain it with a more varied dataset to further refine the model. This dataset would then produce more accurate detection results.

**Primary Project Analyzed and Trained: Yolov5Mode** - In this project the research team developed a model to detect a mask on a person. We piggybacked off this project, but modified and refined the model to detect cracks in images. We trained a model using a new dataset with different labels. We found a varied dataset of images with various sizes, shapes, and colors of cracks on different surfaces. 

Secondary Project Analyzed: While not the primary focus of the project, we did compare our enhanced trained Yolo5 Model with a model trained using the DeepCrack project, which can be found here: https://github.com/qinnzou/DeepCrack 

