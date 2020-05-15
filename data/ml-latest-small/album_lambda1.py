import json
import logging
import boto3
from botocore.vendored import requests
from requests_aws4auth import AWS4Auth

service = 'es'
credentials = boto3.Session().get_credentials()
print('CRED',credentials)
#awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)


logger = logging.getLogger()
logger.setLevel(logging.INFO)





def lambda_handler(event, context):
    # TODO implement
    logger.info(event)
    bucket_name,image_key,event_time = get_info(event)
    logger.info("bucket name:{}   image key:{}   time:{}".format(bucket_name,image_key,event_time))
    
    img_idx = {'S3Object': {'Bucket': bucket_name, 'Name': image_key}} #input for rekognition json format
    logger.info("imgidx:{}".format(img_idx))
    
    img_labels = get_label(img_idx) #list of labels
    logger.info("labels:{}".format(img_labels))
    
    img_json = make_json(bucket_name,image_key,event_time,img_labels) #json to upload to ES
    logger.info("json:{}".format(img_json))
    
    
    ## try to upload json to ELASTIC SEARCH
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }



def get_info(event):
    record = event['Records'][-1]
    bucket_name = record['s3']['bucket']['name']
    image_key = record['s3']['object']['key'].replace('+', ' ')
    event_time = record['eventTime']
    return bucket_name,image_key,event_time
    
def get_label(img_idx):
    rekog_client = boto3.client('rekognition')
    rekog_response = rekog_client.detect_labels(Image = img_idx)
    logger.info("get label finished")
    labels = rekog_response['Labels']
    logger.info(labels)
    img_labels = []
    for label in labels:
        img_labels.append(label["Name"]) 
    return img_labels

def make_json(bucket_name,image_key,event_time,img_labels):
    json = {
        "objectKey": image_key,
        "bucket": bucket_name,
        "createdTimestamp": event_time,
        "labels": img_labels
        
    }
    return json