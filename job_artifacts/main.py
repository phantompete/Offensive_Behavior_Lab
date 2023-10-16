
### input variables for start of workshop
####################################
#################################### CHANGE THE BELOW PARAMETERS
####################################
####################################

password = "Welcome*12345"                                # Database password for user name
wallet_name = "<WALLET_NAME"                               # The name of wallet or database, like "DB202112101358", excluding "Wallet_" and excluding ".zip"
compartment_id_input = "<COMPARTMENT_OCID>"                # OCID of the comparment














####################################
#################################### DO NOT CHANGE THE BELOW.
####################################
####################################

wallet_storage_directory = './wallet'           

#imports
import os
import numpy as np
import pandas as pd 
from deepface import DeepFace
import uuid
import glob      
import ocifs
import base64
import io
import matplotlib.pyplot as plt
import cv2
import fsspec
from PIL import Image
from io import BytesIO
from pytube import YouTube
import numpy as np
import pandas as pd 
import oci
import json
import ocifs
import time
from oci.object_storage import ObjectStorageClient
from oci.ai_language import AIServiceLanguageClient
from oci.ai_language.models import DetectLanguageKeyPhrasesDetails
from oci.ai_language.models import DetectLanguageSentimentsDetails
from oci.ai_speech import AIServiceSpeechClient
from oci.ai_speech.models import TranscriptionModelDetails
from oci.ai_speech.models import ObjectLocation
from oci.ai_speech.models import ObjectListInlineInputLocation
from oci.ai_speech.models import OutputLocation
from oci.ai_speech.models import CreateTranscriptionJobDetails
import sqlalchemy
from sqlalchemy import create_engine
import cx_Oracle
import ads

#import custom py files
from sub_packages.video_only import input_youtube_video
from sub_packages.video_only import input_profile_image
from sub_packages.video_only import predict_video
from sub_packages.audio_only import input_youtube_video_audio
from sub_packages.audio_only import audio_to_object_storage
from sub_packages.audio_only import run_speech_model
from sub_packages.audio_only import parse_results
from sub_packages.audio_only import run_language_models
from sub_packages.audio_only import predict_audio
from sub_packages.credentials import create_uri
from sub_packages.roberta import preprocess
from sub_packages.roberta import roberta_base
from sub_packages.check_table import check_table

print("Main imports done")

#fetch environment variables from APEX input. The below are default values.
TYPE_OF_ANALYSIS = os.environ.get("TYPE_OF_ANALYSIS", "both")
YOUTUBE_URL = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/shorts/ugwUcdtygok")
MAIN_BUCKET_NAME = os.environ.get("MAIN_BUCKET_NAME", "West_BP")
NAMESPACE_NAME = os.environ.get("NAMESPACE_NAME", "frqap2zhtzbe")
#SUB_BUCKET_NAME = os.environ.get("SUB_BUCKET_NAME", "las_vegas")
SCHEMA_NAME = os.environ.get("SCHEMA_NAME", "test")

#Get job run ocid
JOB_RUN_OCID_KEY = "JOB_RUN_OCID"
job_run_ocid = os.environ.get(JOB_RUN_OCID_KEY, "UNDEFINED")

print("Type of analysis is " + TYPE_OF_ANALYSIS)
print("Youtube URL is " + YOUTUBE_URL)
print("Bucket name is " + MAIN_BUCKET_NAME)
print("Schema name " + SCHEMA_NAME)

#get full bucket, namespace, sub_bucket directory
full_bucket_name = "oci://"+MAIN_BUCKET_NAME+"@"+NAMESPACE_NAME+"/"+SCHEMA_NAME+"/"

#route type of analysis

if TYPE_OF_ANALYSIS == 'video_only':
    output_in_screen, seconds_in_screen, total_seconds_video_analyzed = predict_video(full_bucket_name, YOUTUBE_URL, SCHEMA_NAME)
    transcription = 'no_transcription'
    key_phrases_string = 'no_result'
    sentiment_result_string = 'no_result'
    neg_aspects = 0
    non_offensive = 0   #default values
    offensive = 0
    non_hate = 0
    hate = 0
    
elif TYPE_OF_ANALYSIS == 'audio_only':
    transcription, key_phrases_string, sentiment_result_string, neg_aspects = predict_audio(YOUTUBE_URL, MAIN_BUCKET_NAME, NAMESPACE_NAME, compartment_id_input)
    new_text = preprocess(transcription)
    non_offensive, offensive, non_hate, hate = roberta_base(new_text)
    output_in_screen = "no_video"
    seconds_in_screen = 0
    total_seconds_video_analyzed = 0

elif TYPE_OF_ANALYSIS == 'both':
    output_in_screen, seconds_in_screen, total_seconds_video_analyzed = predict_video(full_bucket_name, YOUTUBE_URL, SCHEMA_NAME)  #video
    transcription, key_phrases_string, sentiment_result_string, neg_aspects = predict_audio(YOUTUBE_URL, MAIN_BUCKET_NAME, NAMESPACE_NAME, compartment_id_input)  #audio
    new_text = preprocess(transcription)   # cleans text for input roberta model
    non_offensive, offensive, non_hate, hate = roberta_base(new_text)   #roberta model
    
else: #default values if job fails
    output_in_screen = "unclear"
    seconds_in_screen = 0
    total_seconds_video_analyzed = 0
    transcription = 'no_transcription'
    key_phrases_string = 'no_result'
    sentiment_result_string = 'no_result'
    neg_aspects = 0
    non_offensive = 1   #default values
    offensive = 1
    non_hate = 1
    hate = 1

#change floats to integers
non_offensive_int = int(round((non_offensive * 100),0))
offensive_int = int(round((offensive * 100),0))
non_hate_int = int(round((non_hate * 100),0))
hate_int = int(round((hate * 100),0))
    
    
#Construct all variables in a single line as PD dataframe
data = [[job_run_ocid, TYPE_OF_ANALYSIS, output_in_screen, seconds_in_screen, total_seconds_video_analyzed, 
         transcription, key_phrases_string, sentiment_result_string, neg_aspects, 
        non_offensive_int, offensive_int, non_hate_int, hate_int]]

df_resultsx = pd.DataFrame(data, columns = ['job_run_ocid', 'type_of_analysis', 'output_in_screen', 'seconds_in_screen', 'total_seconds_video_analyzed',
                                           'transcription', 'key_phrases_string', 'sentiment_result_string', 'neg_aspects',
                                           'non_offensive_int', 'offensive_int', 'non_hate_int', 'hate_int'])



print("End of audio and video. Results are printed below")
print(seconds_in_screen)
print(total_seconds_video_analyzed)
print(transcription)
print(key_phrases_string)
print(sentiment_result_string)
    
#### create uri using credentials.py and create engine
engine, wallet_filename = create_uri(SCHEMA_NAME, password, wallet_name, wallet_storage_directory)                                                                #------------------xxxxxxxxxxxxxxxxxx CHANGE user_name in notebook, SCHEMA_NAME in Job
#engine = create_uri(user_name, password, wallet_name, wallet_storage_directory)   
print(engine)

#check whether table exists already or not. Will be 'replace' or 'append'
table_status = check_table(SCHEMA_NAME, password, wallet_name, wallet_filename)

## push results to database
df_resultsx.to_sql('ocw_run_results', con=engine, index=False, if_exists=table_status, dtype={
    
            'job_run_ocid': sqlalchemy.types.NVARCHAR(length=500),    ## from video
            'type_of_analysis': sqlalchemy.types.NVARCHAR(length=500),
            'output_in_screen': sqlalchemy.types.NVARCHAR(length=500),
            'seconds_in_screen': sqlalchemy.types.INTEGER(),
            'total_seconds_video_analyzed': sqlalchemy.types.INTEGER(),
    
            'transcription': sqlalchemy.types.NVARCHAR(length=500),     ## from audio
            'key_phrases_string': sqlalchemy.types.NVARCHAR(length=500),
            'sentiment_result_string': sqlalchemy.types.NVARCHAR(length=500),
            'neg_aspects': sqlalchemy.types.INTEGER(),
    
            'non_offensive_int': sqlalchemy.types.INTEGER(), #roberta
            'offensive_int': sqlalchemy.types.INTEGER(), 
            'non_hate_int': sqlalchemy.types.INTEGER(), 
            'hate_int': sqlalchemy.types.INTEGER() })


print("Table replaced and updated with results")
print("----------- The end -----------------")
