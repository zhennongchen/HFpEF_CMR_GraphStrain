import sys
sys.path.append("/workspace/Documents")
import numpy as np
import os
from sklearn.linear_model import ElasticNetCV, LogisticRegression
import HFpEF_CMR_GraphStrain.functions_collection as ff

class Load:
    def __init__(self, main_path):
        self.main_path = main_path


    def load_strain(self, label_sheet, aha_num = 16, tf_num = 25):
    
        X_Ecc = np.zeros([label_sheet.shape[0], aha_num,tf_num])
        X_Err = np.zeros([label_sheet.shape[0], aha_num-4,tf_num])


        for i in range(0, label_sheet.shape[0]):
            patient_id = ff.XX_to_ID_00XX(label_sheet.iloc[i, 0])
            # print(patient_id)
            Ecc_aha = np.load(os.path.join(self.main_path, 'strain', patient_id, 'Ecc_aha_sample.npy'))
            Err_aha = np.load(os.path.join(self.main_path, 'strain', patient_id, 'Err_aha_sample.npy'))

            # we don't use Err apical:
            Err_aha = Err_aha[0:(aha_num-4),:]
            
            if i == 0:
                print('shape of Ecc, Err:', Ecc_aha.shape, Err_aha.shape)

            X_Ecc[i,:,:] = Ecc_aha
            X_Err[i,:,:] = Err_aha

        # load data Y
        Y = label_sheet['final_label'].values
        
        return X_Ecc, X_Err, Y


    def load_ehr(self,ehr_sheet):

        # features is all the columns except the first one (OurID)
        features = ehr_sheet.columns.tolist()[1:]

        ehr_sheet = ehr_sheet[features]

        # apply normalization
        # Identify binary columns (only containing 0 and 1)
        binary_cols = [col for col in ehr_sheet.columns if ehr_sheet[col].dropna().nunique() == 2 and set(ehr_sheet[col].dropna().unique()).issubset({0, 1})]

        # Identify non-binary columns (to be normalized)
        non_binary_cols = [col for col in ehr_sheet.columns if col not in binary_cols]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        ehr_sheet[non_binary_cols] = scaler.fit_transform(ehr_sheet[non_binary_cols])

        # # turn it into a numpy array
        ehr_array = ehr_sheet.values
        print('ehr_array.shape:', ehr_array.shape)
        return ehr_array,features
    
    # def select_ehr(self,ehr_array, features,  Y, ehr_sheet):
          
    #     X_train = ehr_array
    #     Y_train = Y
    #     elastic_net = ElasticNetCV(l1_ratio=0.5, cv=6, random_state=1)  # l1_ratio=0.5 means equal mix of L1 and L2
    #     elastic_net.fit(X_train, Y_train)
    #     selected_features = np.where(elastic_net.coef_ != 0)[0]  
    #     print(selected_features, len(selected_features))
    #     print([features[i] for i in selected_features])

    #     ehr_sheet_selected = ehr_sheet.iloc[:,selected_features]
    #     ehr_array_selected = ehr_sheet_selected.values
    #     print(ehr_array_selected.shape)
    #     return ehr_array_selected, selected_features

