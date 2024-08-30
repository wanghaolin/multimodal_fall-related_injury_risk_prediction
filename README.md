## Development of a fall-related injury risk prediction model for inpatients based on the multimodal automated machine learning framework

This repository contains the supplementary materials associated with the implementation of our submission titled "Development of a fall-related injury risk prediction model for inpatients based on the multimodal automated machine learning framework." . It utilizes an automated machine learning tool (the AutoGluon-Tabular framework) enhanced by multimodal approaches.

### Prepare data for model building
FRI data were retrospectively collected from three Chinese tertiary-level general hospitals from January 1, 2013 to April 30, 2023.

1. Read files into a pandas DataFrame. Delete the rows in the table where the essential column contains missing values. For example, drop rows with a missing admission number and obtain the admission number list. 
`patient_ids_fall = df['admisson_number'].dropna().tolist()`
2. Obtain a patient list from multiple files.
`patient_list = list(set(patient_ids_fall) & set(patient_ids_info))`
3. Obtain data subsets based on patient IDs.
`df = df.query('admission_id in @patient_list')`
4. Sort the admission records by the report time of the accident and then drop duplicate records.
```
df.sort_values(axis=0, inplace=True, ascending=True, by=['admission_id','reported_date'])
df = df.drop_duplicates(subset=['admission_id','reported_date'], keep='last')
```

5. Merge multiple files extracted from electronic medical record systems, hospital information systems, adverse event reporting systems, and laboratory information systems by matching the patient IDs. For example, obtain the Barthel index score.
```
barthel = []
pids = df['admission_id'].to_list()
for pid in pids:
    temp = new_df.loc[new_df['admission_id'].isin([pid])]
    if temp.shape[0] > 0:
        barthel.append(temp['TOTAL_SCORE'].to_list()[0])
    else:
        barthel.append(None)

df['barthel_score'] = barthel
```

6. Define the preliminary available predictor variables.
7. Calculate the frequency of medical diagnosis.
```
for index in range(len(pid_list)):
    # patient id
    pid = pid_list[index]
    # date
    d = pdate[index]
    # get patient data by admission_id
    temp = df.loc[df['admission_id'].isin([pid])]
    if temp.shape[0] > 0:
        print(temp.shape)
        cols = []
        for index, row in temp.iterrows():
            feature = row['medical_diagnosis']
            if feature not in cols:
                cols.append(feature)
        # calculate the distribution of the number of diagnosis of patients
        diag_count = len(cols)
        if diag_count in count_dict:
            count_dict[diag_count] += 1
        else:
            count_dict[diag_count] = 1
        # calculate the frequency of medical diagnosis of the dataset
        for c in cols:
            if c in diag_dict:
                diag_dict[c] += 1
            else:
                diag_dict[c] = 1
```
8. The most frequent medical diagnosis are selected as features for this study.
```
selected_diagnosis = []

for t,c in diag_dict.items():
    if c >= threshold and str(t) != 'nan':
        print(t)
        selected_diagnosis.append(t)
```
9.  Medical diagnosis was transformed into binary variables.
```
df[selected_cols] = 0

for index, row in df.iterrows():
    pid = str(row['admission_id'])
    temp = df.loc[df['admission_id'].isin([pid])]
    if temp.shape[0] > 0:
        for index1, row1 in temp.iterrows():
            if row1['medical_diagnosis'] in selected_cols:
                df.loc[index, row1['medical_diagnosis']] = 1
```
10. Surgery record was transformed into binary variables. 
```
for index, row in df.iterrows():
    pid = str(row['admission_id'])
    temp = df.loc[df['admission_id'].isin([pid])]
    if temp.shape[0] > 0:
        for index1, row1 in temp.iterrows():
            df.loc[index, row1['anesthesia_method']] = 1  
            df.loc[index, row1['surgical_name']] = 1  
            df.loc[index, row1['surgical_site']] = 1  
            df.loc[index, row1['surgical grade']] = 1  
```
11. **Temporal Analysis of Medication.** Identify medications used within a specific period of time before the fall. Provisional medical orders and standing medical orders are processed separately.
```
for index, row in df.iterrows():
    timestr = ''
    drug_list = []
    pid = str(row['admission_id'])
    ftime = str(row['occurrence_date'])
    fslot = str(row['occurrence_time_period'])
    ftime_seg = ftime.split(' ')
    fslot_seg = fslot.split('-')
    if len(ftime_seg) == 1 and len(fslot_seg) == 2:
        timestr = ftime_seg[0] + ' ' + fslot_seg[0]
    elif len(ftime_seg) == 2:
        timestr = ftime
    else:
        pass
    t = ''
    if timestr != '':
        if len(timestr) == 19:
            t = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
        elif len(timestr) == 16:
            t = datetime.strptime(timestr, '%Y-%m-%d %H:%M')
        else:
            print(timestr)
    if t != '':
        temp = df.loc[df['admission_id'].isin([pid])]
        if temp.shape[0] > 0:
            for index1, row1 in temp.iterrows():
                ksrq = str(row1['KSRQ'])
                kssj = str(row1['KDSJ'])
                kssj_str = ksrq.split(' ')[0] + ' ' + kssj
                t1 = datetime.strptime(kssj_str, '%Y-%m-%d %H:%M:%S')
                diff = t - t1 
                if threshold > diff.total_seconds() >= 0:
                    print(diff.total_seconds())
                    drug = row1['YZMC'].split('[')[0]
                    if drug not in drug_list:
                        drug_list.append(drug)

    time_list.append(t)
    drug_col.append(str(drug_list))

df['occurrence_time'] = time_list
df['medication'] = drug_col

```
12. Identify commonly used drugs according to their usage frequency. Exclude drugs such as glucose injection.
```
selected_cols = []
for t,c in merge_drug_count.items():
    print(c)
    if c >= threshold and not exclude_drug(t):
        selected_cols.append(t)
```
13. **Temporal Analysis of laboratory results**. The sampling time of laboratory tests should be prior to the occurrence time of the fall event.
```
for index, row in info.iterrows():
    pid = str(row['admission_id'])
    timestr = str(row['time_of_fall_occurrence'])
    if timestr != 'nan':
        t = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
        temp = df.loc[df['admission_id'].isin([pid])]
        if temp.shape[0] > 0:
            for index1, row1 in temp.iterrows():
                t1 = datetime.strptime(str(row1['SAMPLING_TIME']), '%Y-%m-%d %H:%M:%S')
                diff = t - t1
                if diff.total_seconds() >= 0:
                    # higher than normal indicators
                    if row1['QUALITATIVE_RESULT'] == 'h':
                        info.loc[index, 'lab-' + row1['test_item'] + '-H'] = 1
                    # lower than normal indicators
                    if row1['QUALITATIVE_RESULT'] == 'l':
                        info.loc[index, 'lab-' + row1['test_item'] + '-L'] = 1
```
14. **Temporal Analysis of Nursing Records**. For example, extract symptoms from nursing records and transform them into binary variables.
```
for index, row in info.iterrows():
    pid = str(row['admission_id'])
    timestr = str(row['time_of_fall_occurrence'])
    if timestr != 'nan':
        t = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
        temp = df.loc[df['admission_id'].isin([pid])]
        if temp.shape[0] > 0:
            for index1, row1 in temp.iterrows():
                t1 = datetime.strptime(str(row1['PLAN_TIME']), '%Y-%m-%d %H:%M:%S')
                diff = t - t1
                print(diff)
                if threshold >= diff.total_seconds() >= 0:
                    print(pid)
                    info.loc[index, 'diarrhea'] = 1
```
15. Encoding of categorical or ordinal variables.
16. **Text Information Extraction**. Smoking history and drinking history were extracted from the text records of personal history. The text in the original data is in Chinese. Universal Information Extraction model was adopted and enhanced by rules. The information extraction results have undergone manual review.
[Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395) (Lu et al., ACL 2022) 
```
from paddlenlp import Taskflow

schema = ['drinking_history']
ie = Taskflow('information_extraction', schema=schema, model='uie-x-base')
col = []
for i in range(0,len(feature)):
    print(feature[i])
    if not pd.isna(feature[i]):
        if 'Drinking history: Denied' in feature[i]:
            col.append(0)
        elif 'Drinking history: Yes' in feature[i]:
            col.append(1)
        else:
            res = ie(feature[i])
            if 'drinking' in res[0]:
                if 'denied' in res[0]['drinking'][0]['text']:
                    col.append(0)
                elif res[0]['drinking'][0]['text'] == 'history':
                    col.append(0)
                else:
                    col.append(1)
            else:
                col.append(0)
    else:
        col.append(0)
```
17.  **Standardization of diagnosis, surgery and drug names**. A rule-based method is adopted.
18.  Exploratory analysis using ydata_profiling
```
import ydata_profiling

profile = ydata_profiling.ProfileReport(df, explorative=True)
profile.to_file('data.html')
```
19. **Missing-value management**. Delete features with a missing ratio greater than 10% according to the report obtained in the previous step. Then missing values were imputed using the KNN algorithm. Noting that textual features can not be imputed.
```
temp = df[[text_columns]]
df.drop(columns=[text_columns], inplace=True)

X = np.array(df)

from sklearn import impute
imp = impute.KNNImputer()
X = imp.fit_transform(X)

X_df = pd.DataFrame(X, columns=df.columns)
X_df = pd.concat([X_df, temp], axis=1)
```
20. **Outlier detection**. First, to ensure the accuracy of the data, the authors carried out manual inspection of data distribution according to the data profile report. Then an Isolation Forest model was performed to identify samples that may have abnormalities are screened out for further manual review to improve the quality of data.
```
model = IsolationForest(contamination=0.05)
model.fit(X_df)
predictions = model.predict(X_df)
outliers = [X_df.loc[i] for i, p in enumerate(predictions) if p == -1]
print(outliers)
```
### Build the predictive model
1. Identify independent (binary) variables that predominantly take a single value
```
for col in selected_cols:
    if df[col].sum() < threshold:
        df.drop(columns=col, inplace=True)
```
2.  **Data Descriptive Analysis** of continuous and categorical features.
```
for col in [continuous_features]:
    fp.write(col + '    ')
    temp = new_df[col].dropna()
    fp.write(str(np.percentile(temp, (50))))
    fp.write('(' + str(np.percentile(temp, (25))) + ',')
    fp.write(str(np.percentile(temp, (75))) + ')')
    fp.write('\n')

for col in [categorical_features]:
    fp.write(col + '    ')
    count = np.sum(new_df[col])
    fp.write(str(count))
    per = round(1.0*count/new_df.shape[0], 2)
    fp.write('(' + str(per))
    fp.write(')\n')
```
3. **Internal validation strategy** is stratified K-Fold validation.
```
X = np.array(X_df)

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X_df, y)
i = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ...
```
4. Train baseline models (for k-fold validation respectively)
```
X_train = np.load('../temp/X_train_1.npy', allow_pickle=True)
y_train = np.load('../temp/y_train_1.npy', allow_pickle=True)
X_test = np.load('../temp/X_test_1.npy', allow_pickle=True)
y_test = np.load('../temp/y_test_1.npy', allow_pickle=True)

X_train_df = pd.DataFrame(X_train, columns=col)
X_train_df['label'] = y_train[:]
X_test_df = pd.DataFrame(X_test, columns=col)
X_test_df['label'] = y_test[:]

exp = setup(data=X_train_df, target='label', session_id=123, fix_imbalance=True) 

for model in ['gbc','rf','ada','qda','lr','xgboost','svm','et','ridge','lightgbm','lda','dt','nb']:
clf = create_model(model, cross_validation=False)
# tuned_clf = tune_model(clf, optimize = 'Accuracy')
pred = predict_model(clf, data=X_test_df)
pd.DataFrame(pred).to_csv('../temp/pred_' +model + '_1.csv', encoding='gbk')
```
5. **SMOTE** is adopted to solve the problem of imbalanced data distribution.
```
from imblearn.over_sampling import SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
```
6. Train AutoML models using **structured data**
```
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='label').fit(X_train_df, verbosity=2)
```
7. Make predictions using the test set.
```
y_prob = predictor.predict_proba(X_test_df)
```
8. Calculates **feature importance** scores for the given model via permutation importance. The higher the score a feature has, the more important it is to the model’s performance.
```
predictor.feature_importance(X_train_df)
```
9. Perform **Feature selection** through feature importance.
```
temp = pd.DataFrame(list(imp_dict.items()), columns=['feature','importance'])
temp = temp[temp.importance > threshold]
pd.DataFrame(temp['feature'].tolist()).to_csv('../temp/selected_features.txt', header=0, index=False)
```
10. Train multimodal model for **structured data + text**. Train a single neural network that jointly operates on multiple feature types to embed the text, categorical and numeric fields separately and fuse these features across modalities. For instance, Chinese pre-trained BERT with Whole Word Masking was used for text feature.
```
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import FeatureMetadata

feature_metadata = FeatureMetadata.from_df(X_train_df)
feature_metadata.add_special_types({'chief_complaint':['text'], 'past_medical_history':['text'], 'personal_history':['text']}, inplace=True)

predictor = MultiModalPredictor(label='label', verbosity=4, enable_progress_bar=True)
predictor.fit(train_data=X_train_df, hyperparameters={"model.hf_text.checkpoint_name": "hfl/chinese-roberta-wwm-ext-large"}, time_limit=60*60)
```
11.  Train **multi-modal ensemble models**.
```
from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(X_train_df)
feature_metadata.add_special_types({'chief_complaint':['text'], 'past_medical_history':['text'], 'personal_history':['text']}, inplace=True)

predictor = TabularPredictor(label='label', eval_metric='accuracy').fit(
    train_data=X_train_df,
    hyperparameters='multimodal',
    feature_metadata=feature_metadata,
    time_limit=limit,
    )

y_prob = predictor.predict_proba(X_test_df)
```
