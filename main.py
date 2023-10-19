from fastapi import FastAPI
import datetime as dt
from fastapi.responses import RedirectResponse
from .models.message import Message
from .models import chgrisk_input
from .models import chgrisk_output
from .config import config, description
from datetime import datetime
from .chgrisk_s3_imports import s3bucket_chgrisk
from .chgrisk_nlp import DataCleanAndNLP
import pytz
import os
     
app = FastAPI(
    title=config.title,
    description=description,
    version=config.version,
    terms_of_service=config.terms_of_service,
    openapi_tags=config.tags_metadata,
    contact={
        "name": "GT Applied Analytics",
        "url": "https://prudentialus.sharepoint.com/sites/DIY0292/SitePages/The-Data-Dogs.aspx",
        "email": "gt.applied.analytics@prudential.com",
    }
)

@app.get("/", tags=["Home"])
async def get_home():
    """Get the Home message"""
    return {"msg": "The API is healthy!"}

@app.get("/message", tags=["Test messages"])
async def get_message():
    """Get the Welcome message"""
    return {"msg": "The FastAPI server is working! 🚀"}

@app.post("/message", tags=["Test messages"])
async def post_message(message: Message):
    """Post a welcome message"""
    return {"msg": message.msg}

@app.post("/eval_chg_risk", tags=["Evaluating risk"], response_model=chgrisk_output.ChgRiskOutput)
async def post_change(chgrisk_input: chgrisk_input.ChgRiskInput):
    """POST a CHG to be evaluated"""
    # Call the s3 file to class and create and object for it
    print(f"S3 LOAD FUNCTION CALLED @  : {dt.datetime.now()}")
    getDataFromS3 = s3bucket_chgrisk()
    # Download encoder pkl file
    print(f"CHGRISK_ENCODER PICKLE LOAD STARTED @  : {dt.datetime.now()}")
    pathForDataEncoder = getDataFromS3.get_s3_chgrisk_encoder()
    print(f"CHGRISK_ENCODER PICKLE LOAD ENDED @  : {dt.datetime.now()}")
    # Download model pkl file
    print(f"CHGRISK_MODEL PICKLE LOAD STARTED @  : {dt.datetime.now()}")
    pathForDataModel = getDataFromS3.get_s3_chgrisk_model()
    print(f"CHGRISK_MODEL PICKLE LOAD ENDED @  : {dt.datetime.now()}")
    # Call to Process data and run it against CHANGE RISK AI MODEL
    print(f"CHANGE.RISK.AI.MODEL RISK CALCULATION STARTED @  : {dt.datetime.now()}")
    print(f'CHANGE_TICKET_INFO : {chgrisk_input}')
    scoring = DataCleanAndNLP(pathForDataModel,pathForDataEncoder,chgrisk_input)
    response_data = chgrisk_output.ChgRiskOutput(
        chgrisk_class_prediction =  scoring['chgrisk_class_prediction'],
        chgrisk_model_probability = scoring['chgrisk_model_probability'],
        chgrisk_scoring_datetime = scoring['chgrisk_scoring_datetime'],
        chgrisk_model_version = scoring['chgrisk_model_version']
    )
    print(f"CHANGE.RISK.AI.MODEL RISK CALCULATION ENDED @  : {dt.datetime.now()}")
    print(f"API CALL ENDED @  : {dt.datetime.now()}")
    return response_data

    
  

