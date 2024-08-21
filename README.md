# DSB-Capstone

## **Executive Summary**

### **Introduction**

Due to their complex nature and varying types, brain tumors pose significant challenges in medical diagnostics. The key to improved outcomes and deciding appropriate treatment is early and accurate classification of brain tumors. Neural Networks, specifically Convolutional Neural Networks (CNNs), have proven themselves valuable in automating the classification of medical images. 

This project focuses on leveraging CNNs to classify brain tumors and developing a strong model able to accurately identify and categorize classes of MRI brain images. The diverse dataset used to train these models will aid in achieving high classification accuracy and reliability. Not only can this boost diagnostic workflow and efficiency but also provide tools for radiologists and clinicians in making informed decisions regarding patient care and leading to more accurate detection of brain tumors. 

### **Methods**

In order to process the data, python was used with a multitude of imported libraries: pandas, numpy, matplotlib, tensorflow, sklearn and keras. The data came from a Kaggle data set organized by train and validation. Within each set, images were in four tumor classes: glioma, meningioma, no tumor and pituitary. A total of 4,741 training images and 516 validation images were used. 

#### **Data Cleaning**

In order to standardize all the images as they were being read in, they went through a preprocessing step. Images were all resized to 256 x 256 then, once the train test split was instantiated, X_train and X_test were standardized by dividing by 255. The labels (y_train and y_test) were transformed to categorical values. 

#### **Instantiating the Models***

##### **CNN + MaxPooling & Dropout (and l2 regularization)**

A keras Sequential model was instantiated with 2 Conv2D layers coupled with MaxPooling2D and Dropout. The sequence was a follows:

	Conv2D(32, 5, activation=’relu’, input_shape(256, 256, 3)),
	MaxPooling2D(2),
	Dropout(0.2),
	Conv2d(64, 5, activation=’relu’),
	MaxPooling2D(2),
	Dropout(0.25),
	Flatten(),
	Dense(128, activation=’relu’),
	Dropout(0.25),
	Dense(4, activation=’softmax’)

With this sequence, we had 30,536,722 trainable parameters. The model was compiled using a loss of categorical crossentropy, optimizer of adam, and accuracy for metrics. Finally, this model was fit with a batch size of 64 and 20 epochs. 

An additional model was run with an l2 regularization of 0.001. The sequence, compile and fit was the same as above. 

#### **CNN + Transfer Learning (Xception)**

Xception is a transfer learning model that is 71 layers deep. It is specifically used for image classification. While it is relatively simple, it has been shown to perform well in various image classifications. In this instance, the model looked as such:

    base_model = Xception(weights=’imagenet’, include_top=False, input_shape(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=’relu’)(x)
    predictions = Dense(len(class_names), activation=’softmax’)(x)
    model = keras.Model(input=base_model.input, outputs=predictions)
    
With this model, we had a total of 22,963,756 parameters but only 2,102,276 parameters were trainable. The model was compiled using a loss of categorical crossentropy, optimizer of adam, and accuracy for metrics. Finally, this model was fit with a batch size of 64 and 20 epochs.

### **Results**

#### **Baseline**

In the training data, the number of images in each class was as such: 1449 Meningioma, 1424 Pituitary, 1153 Glioma, and 711 No Tumor. 

In the testing data, the number of images in each class was as such: 140 Meningioma, 136 Pituitary, 136 Glioma, and 100 No Tumor.

#### **CNN + MaxPooling & Dropout**

The overall accuracy of the **custom CNN (without l2 regularization)** was 0.93.  

    Classification Metrics:

|  | Precision | Recall | F1-Score |
|----------|----------|----------|----------|
| Glioma | 0.90 | 0.90 | 0.90 |
| Meningioma | 0.87 | 0.95 | 0.91 |
| No Tumor | 0.98 | 0.88 | 0.93 |
| Pituitary | 0.99 | 0.97 | 0.98 |

    Confusion Matrix:

|  | Pred Glioma | Pred Meningioma | Pred No Tumor | Pred Pituitary |
|----------|----------|----------|----------| --------- |
| Actual Glioma | 122 | 14 | 0 | 0 |
| Actual Meningioma | 6 | 133 | 1 | 0 |
| Actual No Tumor | 5 | 4 | 88 | 2 |
| Actual Pituitary | 2 | 1 | 1 | 132 |

The overall accuracy of the **custom CNN model (with l2 regularization)** was 0.93.

    Classification Metrics:

|  | Precision | Recall | F1-Score |
|----------|----------|----------|----------|
| Glioma | 0.91 | 0.93 | 0.92 |
| Meningioma | 0.87 | 0.92 | 0.90 |
| No Tumor | 0.98 | 0.86 | 0.91 |
| Pituitary | 0.99 | 0.99 | 0.99 |

    Confusion Matrix:

|  | Pred Glioma | Pred Meningioma | Pred No Tumor | Pred Pituitary |
|----------|----------|----------|----------| --------- |
| Actual Glioma | 127 | 9 | 0 | 0 |
| Actual Meningioma | 9 | 139 | 2 | 0 |
| Actual No Tumor | 3 | 9 | 86 | 2 |
| Actual Pituitary | 1 | 1 | 1 | 134 |


#### **CNN + Transfer Learning (Xception)**
The overall accuracy of the **CNN with Xception architecture** was 0.95.

    Classification Metrics:
    
|  | Precision | Recall | F1-Score |
|----------|----------|----------|----------|
| Glioma | 0.98 | 0.90 | 0.94 |
| Meningioma | 0.86 | 0.97 | 0.91 |
| No Tumor | 1.00 | 0.95 | 0.97 |
| Pituitary | 0.98 | 0.97 | 0.97 |

    Confusion Matrix:

|  | Pred Glioma | Pred Meningioma | Pred No Tumor | Pred Pituitary |
|----------|----------|----------|----------| --------- |
| Actual Glioma | 122 | 13 | 0 | 1 |
| Actual Meningioma | 4 | 136 | 0 | 2 |
| Actual No Tumor | 0 | 5 | 95 | 2 |
| Actual Pituitary | 0 | 4 | 0 | 132 |

### **Discussion/Conclusions/Next Steps**

This project involved developing CNN models for classifying brain MRI images, which is crucial for several reasons. Firstly, these models enhance diagnostic accuracy and efficiency, enabling early detection of brain tumors and significantly improving patient prognosis. Early detection offers more treatment options and enhances overall outcomes by allowing patients to explore various treatment choices. Additionally, automation in detecting and classifying brain tumors means healthcare professionals can focus on more complex issues, as automated systems can process and analyze images faster and more accurately than human radiologists.

To determine the most effective models, two approaches were employed: a custom CNN model with traditional layers (max pooling, dropout, and L2 regularization) and a transfer learning model using the Xception architecture. The custom CNN model demonstrated high accuracy and strong classification metrics, with F1-scores at or above 0.90 for all classes and an overall accuracy of 0.93. However, it consistently underperformed compared to the Xception model, which utilizes pre-trained knowledge from large-scale image datasets. The Xception model achieved F1-scores at or above 0.91 for all classes and an overall accuracy of 0.95, highlighting the benefits of transfer learning. Future research may explore other transfer learning models to further optimize efficiency and scalability, with continuous feedback from medical professionals being essential for refining and adapting the model to clinical needs.

Advancing medical technology is another significant outcome of this project. The insights gained from these models drive innovation in medical imaging and open new avenues for research and the exploration of novel approaches in medical image analysis. Additionally, these models serve as educational tools for medical professionals, helping them interpret complex imaging data more effectively. From an ethical and societal perspective, the integration of advanced diagnostic tools in underserved areas has the potential to reduce healthcare disparities and empower patients with precise health information, ultimately contributing to more equitable health outcomes.

However, there are limitations to consider. The computational resources required for models like Xception, such as powerful GPUs and substantial memory, may not be available in all institutions. Additionally, CNNs are often criticized for their lack of interpretability, which can be a barrier for medical professionals and patients who value understanding the decision-making process. For machine learning models to be trusted in clinical settings, they must demonstrate high accuracy, reliability, and transparency. Integrating CNN models into clinical workflows may also require complex adjustments to ensure seamless operation.

In conclusion, the CNN model with the Xception architecture demonstrated superior accuracy and performance in classifying brain MRI images. This project underscores the potential of machine learning models to advance automated medical image analysis and improve diagnostic processes in clinical settings. The ultimate goal of healthcare automation is to enhance patient care and support medical professionals in decision-making, and continued research and development in this area will be invaluable for achieving these objectives.

