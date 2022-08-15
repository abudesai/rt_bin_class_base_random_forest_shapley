import numpy as np, pandas as pd
import os, sys
import json
from shap import Explainer


import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]  
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 5
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            try: 
                self.preprocessor = pipeline.load_preprocessor(self.model_path)
                return self.preprocessor
            except: 
                print(f'Could not load preprocessor from {self.model_path}. Did you train the model first?')
                return None
        else: return self.preprocessor
    
    def _get_model(self): 
        if self.model is None: 
            try: 
                self.model = classifier.load_model(self.model_path)
                return self.model
            except: 
                print(f'Could not load model from {self.model_path}. Did you train the model first?')
                return None
        else: return self.model
        
    
    def _get_predictions(self, data): 
        '''
        Returns the predicted class
        ''' 
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X = proc_data['X'].astype(np.float)        
        # make predictions
        preds = model.predict( pred_X )
        return preds    
    
    
    def predict_proba(self, data):  
        '''
        Returns predicted probabilities of each class
        '''         
        preds = self._get_predictions(data)
        # get class names (labels)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)       
        # get the name for the id field
        
        # return te prediction df with the id and class probability fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df[class_names[0]] = 1 - preds   
        preds_df[class_names[1]] = preds  
        #print(preds_df) 
        
        return preds_df    
    
    
    def _get_target_class_proba(self, X): 
        '''
        Returns predicted probability of the target class
        ''' 
        model = self._get_model()
        preds= model.predict_proba(X)
        return preds[:,1]
    
    
    def explain_local(self, data): 
        
        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f'''Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations.'''
            print(msg)
        
        preprocessor = self._get_preprocessor()        
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))         
        # ------------------------------------------------------------------------------
        # original class predictions 
        
        model = self._get_model()
    
        pred_X = proc_data['X'].astype(np.float)   
        ids = proc_data['ids']
        
        pred_classes = model.predict(pred_X)
        pred_target_class_prob = model.predict_proba(pred_X)[:, 1]        
        
        # ------------------------------------------------------------------------------
        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")   
        #create the shapley explainer 
        mask = np.zeros_like(pred_X)
        explainer = Explainer(self._get_target_class_proba, mask, seed=1)
        # Get local explanations        
        shap_values = explainer(pred_X)
        
        # ------------------------------------------------------------------------------
        # create pd dataframe of explanation scores
        N = pred_X.shape[0]
        explanations = []
        for i in range(N):
            samle_expl_dict = {}
            samle_expl_dict[self.id_field_name] = ids[i]
            samle_expl_dict['predicted_class'] = pred_classes[i]
            samle_expl_dict['predicted_class_prob'] = pred_target_class_prob[i]
            samle_expl_dict['baseline_prob'] = shap_values.base_values[i]
            
            feature_impacts = {}
            for f_num, feature in enumerate(shap_values.feature_names):
                feature_impacts[feature] = round(shap_values.values[i][f_num],4)
            
            samle_expl_dict["feature_impacts"] = feature_impacts
            explanations.append(samle_expl_dict)
            
        # ------------------------------------------------------ 
        '''
        To plot the shapley values:
        you can only plot one sample at a time. 
        if you want to plot all samples. create a loop and use the index (sample_idx)
        '''       
        # sample_idx = 4
        # shap_values.base_values = shap_values.base_values[sample_idx]
        # shap_values.values = shap_values.values[sample_idx]
        # shap_values.data = shap_values.data[sample_idx]        
        # shap.plots.waterfall(shap_values)
        # ------------------------------------------------------  
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations

