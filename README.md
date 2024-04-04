# AMLSII_23-24_SN23077662
Applied Machine Learning Systems II(ELEC0135) final assignment
### Description
This project provides a solution to the Kaggle competition: "Cassava Leaf Disease Classification" [https://www.kaggle.com/competitions/cassava-leaf-disease-classification]   
 This repository contains four models based on a custom CNN model and transfer learning with the EfficientNetB3 model. Techniques such as Batch Normalization, Dropout, Data Augmentation, ensemble model and Class Weight are tried.  
 
| Model                                                    | Test Accuracy |
|----------------------------------------------------------|---------------|
| Base CNN                                                 | 0.81          |
| CNN + Image augmentation + class_weight                  | 0.77          |
| EfficientNetB3 transfer learning + Image augmentation + class_weight | 0.89 |
| Ensemble model of baseline CNN and EfficientNetB3 transfer learning model | 0.88 |

### Python libraries required

 - numpy
 - scikit-learn
 - tensorflow
 - keras
 - pandas
 - seaborn
 - openCV(cv2)
 - Pillow
 - matplotlib

### Program structure

-- AMLS_23-24_SN23077662
```
- Datasets
- Results
- Models
  - EfficientNetB3.h5 (pre-trained CNN model)
  - baseCNN.h5
  - improvedCNN.h5
- .gitattributes
- main.py (file to run the whole project)
- cassava_classification.py
- baseCNN.ipynb
- improvedCNN.ipynb
- EfficientNet.ipynb
- README.md
```
- `Datasets/`:Please download and unzip the dataset here(please contain folder:"cassava-leaf-disease-classification")
- `results/`: Empty folder, save the model if trained manually
- `Models`: Contains three pre-trained models
- `.gitattributes`: Git attributes file for large file(Git lfs).
- `cassava_classification`: Contains functions for the whole project used by main.py
- `main.py`: The main script for executing the whole project. 
- `baseCNN.ipynb`: Kaggle notebook shows the model training process for custom CNN
- `improvedCNN.ipynb`: Kaggle notebook shows the model training process for CNN+image augmentation+class_weight
- `EfficientNet.ipynb`: Kaggle notebook shows the model training process for EfficientNetB3 transfer learning model
- `README.md`: The README file for the project.

### Program run instruction
1. Make sure all required libraries are installed in the local environment
2. Clone the repository.
3. Download the dataset on the Kaggle competition data page(link above)
4. Unzip the dataset (contains folder: "cassava-leaf-disease-classification") in 'Datasets' folder.
5. Run "main.py" which loads pre-trained data and evaluates them(change the first param "train" to True to begin training instead of loading)
6. Detailed training processes are shown in notebooks for the pre-trained models. (recommend!)

### Before running
- The program default loads the pre-trained model rather than training from scratch which may take more than 20 hours on GPU P100 provided by Kaggle. 
- You can change the first param "train" to True to begin training instead of loading pre-trained models.
- The Kaggle competition does not provide test images and labels so the training data are used and split for that with a ratio of 0.1.
- Training the models on different platforms and environments may lead to little difference in results.
- For GPU requirements, the notebook is created and running on Kaggle based on the competition Python environment pin to 2020. Please make sure to use this environment to avoid conflict. You could copy a baseline competition code notebook and copy the script into that notebook to use the environment. 
