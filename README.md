**Lunar**

**Summary**

Lunar predicts the stress level of an individual for attending future events based on the calendar. 

1. Raw Input: (1) events from calendar and stress level (similar to train.csv) are used for training the model, (2) unlabeled events are the test data (similar to test.csv)

2. Data Cleansing/Manipulation: feature engineering +  tokenizing the summary column in the calendar and check the presence of keywords for each influence + applying Principal Component Analysis (PCA) 

3. Created Features: managers-key, co-workers-key, customers-key, meetings-key, tasks-key, thoughts-key, finances-key, health-key, spouse/partner-key, children-key, parents-key, friends-key, body-key, user_id, duration, is_online, num_participants, r_start, angle_start, r_end, angle_end, duration, morning

4. ML Methods: Gradient Boosting 

5. The sequence of Methods Applied:

(A) Reading input from train.csv, initial data processing, and cleaning to create the raw features in a readable format, (B) Feature engineering, (C) Applying SMOTE for oversampling (D) Applying PCA, (E) Training the model, (F) Reading input from test.csv and initial data processing and cleaning to create the raw features in a readable format, (G) Feature engineering, (H) Applying PCA, (I) Generating labels for new data.

6. Outputs: stress level values that are integer in the range from 1 to 5.

7. Evaluation: the average accuracy of training and testings using 5-fold cross-validation is 100% and 80%, respectively.

**Requirements** 

Python 3, pandas, NumPy, Sklearn, iso8601, imblearn

The main folder contains input, keywords, output, and pickle folders.

The input folder should contain train.csv and test.csv

The keywords folder contains text files. Each text file has a list of keywords related to each influence. The mapping of these keywords to each influence could be found in this [folder](https://drive.google.com/drive/folders/1G7lQnJOdeR7-2kd86ijnQb9Q7qY96jD9?usp=sharing).

**Running**

Go to the main folder and run the file using the following command:
 
python main.py 

You will find 4 output files as follows:

(1) predicted_label.csv, (2) time_stamp_info.p, (3) pca.p, and (4) trained_model.p

time_stamp_info.p contains parameters for transforming the timestamps.

pca.p and trained_model.p are Pickle files that save the parameters of PCA and trained classifier.
