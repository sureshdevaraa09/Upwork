import spacy
import pandas as pd 
import datetime as dt
import pickle
import contractions
from sklearn.preprocessing import OrdinalEncoder
import sklearn
import dill
import pytz
import numpy as np

def DataCleanAndNLP(pathForDataModel,pathForDataEncoder,chgrisk_input):
    chgrisk_model_pkl = open(pathForDataModel, 'rb')
    chgrisk_model = dill.load(chgrisk_model_pkl)
    
    # Input Data for model scoring
    print('CREATING DATAFRAME')
    data = [[chgrisk_input.number,chgrisk_input.contact_type,chgrisk_input.risk_impact_analysis, chgrisk_input.dv_u_ci_affect, chgrisk_input.dv_u_ci_impact,
    chgrisk_input.dv_u_ci_users, chgrisk_input.dv_category, chgrisk_input.dv_assignment_group, chgrisk_input.short_description,chgrisk_input.start_date, 
    chgrisk_input.end_date, chgrisk_input.dv_cmdb_ci]]
    df = pd.DataFrame(data, columns=['number', 'contact_type','risk_impact_analysis','dv_u_ci_affect','dv_u_ci_impact','dv_u_ci_users','category','assignment_group',
    'short_description_nlp_2', 'start_date', 'end_date', 'dv_cmdb_ci'])
    print('CREATED DATAFRAME')

    # Transform
    print('TRANSFORM STARTED')
    def calculate(val):
        return len(val)
    df1 = pd.Series([chgrisk_input.dv_ci_item])
    print(df1)
    df['chg_aff_count'] = df1.apply(calculate)

    # Creating dv_ci_item
    new_DF = df
    for i in range(0,len(chgrisk_input.dv_ci_item)-1):
        new_DF.loc[len(new_DF.index)] = list(new_DF.iloc[0])
    new_DF['dv_ci_item_nlp_2'] = chgrisk_input.dv_ci_item
    df = new_DF
    
    def check_for_val(dat):
        if "prod" in dat.lower():
            return "PROD"
        if "qa" in dat.lower():
            return "QA"
        if "stage" in dat.lower():
            return "STAGE"
        if "test" in dat.lower():
            return "TEST"
        if "dev" in dat.lower():
            return "DEV"
        if "uat" in dat.lower():
            return "UAT"
        return "PROD"

    # Creating ci_type
    df['ci_type'] = df['dv_cmdb_ci'].apply(check_for_val)
    df['chg_hours'] = df['end_date'] - df['start_date']
    df['chg_hours'] = df['chg_hours'].dt.components['hours']
    df_raw = df.copy()
    print('TRANSFORM ENDED')

    df = df [['ci_type','contact_type','risk_impact_analysis','dv_u_ci_affect','dv_u_ci_impact','dv_u_ci_users',
    'category','assignment_group','short_description_nlp_2','dv_ci_item_nlp_2','chg_aff_count','chg_hours']]
    
    # Encoding File Import & Encoding the Input Data
    print('ENCODING_STARTED')
    with open(pathForDataEncoder, 'rb') as file:
        encoder = dill.load(file)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, min_frequency=0.01) 
    encoder.fit(df)
    df = encoder.transform(df)
    print('ENCODING_ENDED')
    
    # Scoring
    print('SCORING_STARTED')
    pred = chgrisk_model.predict(df)
    df_raw['chgrisk_class_prediction'] = pred
    df_raw['chgrisk_model_probability'] = [max(i) for i in chgrisk_model.predict_proba(df).round(3)]
    df_raw['chgrisk_scoring_datetime'] = dt.datetime.now(pytz.utc)
    df_raw['chgrisk_model_version'] = 'P_V1.0'
    currentDateTime = dt.datetime.now(pytz.utc).strftime("%m-%d-%Y %H-%M-%S %p")
    df_raw['num_row'] = (df_raw.sort_values(by=['chgrisk_class_prediction','chgrisk_model_probability'],ascending=[False,False]).groupby(['number'], sort=True).cumcount().add(1))
    df_raw = df_raw.loc[df_raw['num_row'] == 1]
    dictdata = dict()
    dictdata['chgrisk_class_prediction']=df_raw['chgrisk_class_prediction'].values[0]
    dictdata['chgrisk_model_probability']=df_raw['chgrisk_model_probability'].values[0]
    dictdata['chgrisk_scoring_datetime']=df_raw['chgrisk_scoring_datetime'].values[0]
    dictdata['chgrisk_model_version']=df_raw['chgrisk_model_version'].values[0]
    print('SCORING_ENDED')
    return dictdata
 