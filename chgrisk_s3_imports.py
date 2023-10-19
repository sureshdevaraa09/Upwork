import boto3
import time , os
from gtaa.context_helper import context_helper
import logging

class s3bucket_chgrisk:
    def get_boto_s3(self):
        logger = logging.getLogger("ChgRisk")
        context_util = context_helper(logger)
        aws_access_key_id, aws_secret_access_key, aws_session_token = context_util.get_credentials(logger)
        region_name = 'us-east-1'
        s3 = None
        if aws_access_key_id:
            s3 = boto3.resource('s3',
                aws_session_token=aws_session_token, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)
        else:
            s3 = boto3.resource('s3', region_name=region_name)
        return s3

    # Loading the encoder pickle file
    def get_s3_chgrisk_encoder(self):
        s3 = self.get_boto_s3()
        raw_bucket = 'pruvpcaws144-use1-ctoappliedanalytics-general'
        s3_bucket = s3.Bucket(raw_bucket)
        s3_output_meta = 'analytics/eval_chg_risk/chgrisk_labels_encoder_prod.pkl'
        s3_bucket.download_file(s3_output_meta, 'chgrisk_labels_encoder_prod.pkl')
        return os.path.join(os.getcwd(),'chgrisk_labels_encoder_prod.pkl')
    
    # Loading the model pickle file   
    def get_s3_chgrisk_model(self):
        s3 = self.get_boto_s3()
        raw_bucket = 'pruvpcaws144-use1-ctoappliedanalytics-general'
        s3_bucket = s3.Bucket(raw_bucket)
        s3_output_meta = 'analytics/eval_chg_risk/chgrisk_model_prod.pkl'
        s3_bucket.download_file(s3_output_meta, 'chgrisk_model_prod.pkl')
        return os.path.join(os.getcwd(),'chgrisk_model_prod.pkl')