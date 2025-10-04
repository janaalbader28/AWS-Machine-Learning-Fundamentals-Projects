"""Lambda 1: Get image from S3 and encode it"""

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key =event["s3_key"] 
    bucket =event["s3_bucket"] 
    
    # Download the data from s3 to /tmp/image.png
    boto3.resource('s3').Bucket(bucket).download_file(key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



"""Lambda 2: Send image to SageMaker endpoint"""

import os
import io
import boto3
import json
import base64
# import sagemaker
# from sagemaker.serializers import IdentitySerializer


# setting the  environment variables
ENDPOINT_NAME = 'image-classification-2023-03-10-20-00-10-818'
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])
    
    # Make a prediction:
    predictor = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                    ContentType='application/x-image',
                                      Body=image)
    
    # We return the data back to the Step Function    
    event["inferences"] = json.loads(predictor['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        "body": {
            "image_data": event["body"]['image_data'],
            "s3_bucket": event["body"]['s3_bucket'],
            "s3_key": event["body"]['s3_key'],
            "inferences": event['inferences'],
       }
    }



  
"""Lambda 3: Check if confidence passes threshold"""

 import json


THRESHOLD = .9

def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = event["body"]["inferences"]
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
