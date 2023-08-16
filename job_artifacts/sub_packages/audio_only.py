import os
import numpy as np
import pandas as pd 
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
from ads.model.framework.tensorflow_model import TensorFlowModel
from ads.common.model_metadata import UseCaseType
from ads.common.model_artifact import ModelArtifact
from ads.common.model_export_util import prepare_generic_model
from pytube import YouTube
import oci
import json
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

##########################################################################################################################################
######################################################## Function 1          #############################################################
##########################################################################################################################################

def input_youtube_video_audio(YOUTUBE_URL):
    
    #delete previous videos
    os.system("rm -r /home/datascience/youtube_videos_audio")

    #create a local directory to store the video
    path_input_locally = "/home/datascience/youtube_videos_audio/" 

    try:       
        if not os.path.exists(path_input_locally):         
            os.makedirs(path_input_locally)    

    except OSError: 
        print ('Error: Creating directory for youtube audio locally')
        

    #download file from youtube
    yt = YouTube(YOUTUBE_URL)

    #store in local folder
    stream = yt.streams.get_by_itag(139)  #139 is audio only
    file_name_random = str(uuid.uuid4())
    file_location_local_audio = stream.download(output_path=path_input_locally, filename  = file_name_random + ".mp4")
    
    print("Youtube download for audio only completed and stored in " + str(file_location_local_audio))
    
    return file_location_local_audio

##########################################################################################################################################
######################################################## Function 2          #############################################################
##########################################################################################################################################

def audio_to_object_storage(bucket_name_input, namespace_input, config, name, file_location_local):
    
    #Object Storage Client
    client_object_storage = ObjectStorageClient(config)
    
    #Define bucket and namespace
    bucket_name = bucket_name_input
    namespace = namespace_input

    #Audio to Bucket
    response = client_object_storage.put_object(namespace, bucket_name, name, io.open(file_location_local, 'rb'), content_type='audio/wav')

##########################################################################################################################################
######################################################## Function 3          #############################################################
##########################################################################################################################################

def run_speech_model(bucket_name_input, namespace_input, compartment_id_input, config, name):
    
    # Instantiate Speech Client
    ai_speech_client = AIServiceSpeechClient(config)
    
    # Define Parameters for Transcription Jobs
    job_display_name = "Offensive_Language_Detection"
    job_compartment_id = compartment_id_input
    job_description = "Offensive_Language_Detection"
    bucket_name = bucket_name_input
    namespace = namespace_input
    output_prefix = "speech_out_"
    

    # Define Transcription Job - Model, Data, Input, Outputs
    job_model_details = TranscriptionModelDetails(domain="GENERIC", language_code="en-GB")
    job_object_location = ObjectLocation(namespace_name=namespace, bucket_name=bucket_name,object_names=[name])
    job_input_location = ObjectListInlineInputLocation(location_type="OBJECT_LIST_INLINE_INPUT_LOCATION", object_locations=[job_object_location])
    job_output_location = OutputLocation(namespace_name=namespace, bucket_name=bucket_name, prefix=output_prefix)

    
    # Create Transcription Job with details provided above
    transcription_job_details = CreateTranscriptionJobDetails(display_name=job_display_name,
                                                                compartment_id=job_compartment_id,
                                                                description=job_description,
                                                                model_details=job_model_details,
                                                                input_location=job_input_location,
                                                                output_location=job_output_location)

    
    # Call the AI Speech Service to Create Transcription Job 
    transcription_job = None
    try:
        transcription_job = ai_speech_client.create_transcription_job(create_transcription_job_details=transcription_job_details)
    except Exception as e:
        print(e)
    else:
        print(transcription_job.data.lifecycle_state)
        
    # Pause for 3 Seconds to Allow Job to be Accepted
    time.sleep(3)
    
    # Gets the First Transcription Tasks under given Transcription Job Id then Extracts Info for that Task
    transcription_tasks = None
    try:
        # Get Tasks Under Job
        transcription_tasks = ai_speech_client.list_transcription_tasks(transcription_job.data.id, limit=1)
        
        # Keep Checking until Task is Succeeded
        while transcription_tasks.data.items[0].lifecycle_state != 'SUCCEEDED':
            print('Transcribing in Progress...')
            time.sleep(5)
            transcription_tasks = ai_speech_client.list_transcription_tasks(transcription_job.data.id, limit=1)
            
        # Once Task is Succeeded Extract Task Info
        transcription_task = ai_speech_client.get_transcription_task(transcription_job.data.id, transcription_tasks.data.items[0].id)
        
    except Exception as e:
        print(e)
        
    else:
        print(transcription_tasks.data.items[0].lifecycle_state)
        print(transcription_task.data.output_location.object_names[0])
    
    # Extract Results File Name from Task Info Response
    object_name = transcription_task.data.output_location.object_names[0]
    
    return object_name


##########################################################################################################################################
######################################################## Function 4          #############################################################
##########################################################################################################################################

def parse_results(bucket_name_input, namespace_input, config, object_name):
    
    # Instantiate Object Storage Client
    client = ObjectStorageClient(config)
    
    # Define Parameters
    bucket_name = bucket_name_input
    namespace = namespace_input
    
    # Get Speech Results File from Object Storage
    response = client.get_object(namespace, bucket_name, object_name)
    
    # Decode Results from File
    decoded_resp = json.loads(response.data.content.decode())
    
    # Extract Transcription from Results
    transcription_out = decoded_resp['transcriptions'][0]['transcription']
    print(transcription_out)
    return transcription_out

##########################################################################################################################################
######################################################## Function 5          #############################################################
##########################################################################################################################################

def run_language_models(config, transcription_out):
    
    # Initialize Service Client to Language API
    ai_language_client = AIServiceLanguageClient(config)
    
    
    # Make a REST API Request to AI Language Service to Detect Key Phrases
    language_key_phrases = ai_language_client.detect_language_key_phrases(
        detect_language_key_phrases_details=DetectLanguageKeyPhrasesDetails(text = transcription_out))
    
    # Results List
    key_phrase_results = []
    
    # Extract Language Entities
    formatted_response = language_key_phrases.data.key_phrases
    
    # Iterate through and Store Entites in Results List
    for key_phrase in formatted_response:
        key_phrase_results.append(key_phrase.text)
        
    #merge key phrases extracted from list in one string: 
    key_phrases_string_out = 'input: '
    for key_phrase in key_phrase_results:
        key_phrases_string_out += key_phrase + ', '
        
    ##maximuze input for sentiment and for push to db
    key_phrases_string = key_phrases_string_out[0:498]  #filter on max 500 characters

    # Make a REST API Request to AI Language Service to Detect Sentiments
    language_sentiment_response = ai_language_client.detect_language_sentiments(
        detect_language_sentiments_details=DetectLanguageSentimentsDetails(text = key_phrases_string))
    
    # Results List
    sentiment_results = []
    
    # Extract Language Sentiments
    formatted_response = language_sentiment_response.data.aspects
    
    # Iterate through and Store Aspect Sentiment in Results List
    for aspect in formatted_response:
        sentiment_results.append((aspect.text, aspect.sentiment))

    return key_phrases_string, sentiment_results


##########################################################################################################################################
######################################################## Predict             #############################################################
##########################################################################################################################################


def predict_audio(YOUTUBE_URL, MAIN_BUCKET_NAME, NAMESPACE_NAME, compartment_id_input):
    
    main_bucket_name = MAIN_BUCKET_NAME
    namespace = NAMESPACE_NAME
        
    # Authenticate against OCI 
    config = oci.config.from_file('config', 'DEFAULT')                                                       #### ****** -------- In Job
    #config = oci.config.from_file('/home/datascience/.oci/config', 'DEFAULT')                               #### ****** -------- In notebook
    
    #Function 1. Download YouTube video as recording
    file_location_local_audio = input_youtube_video_audio(YOUTUBE_URL)
    
    # Send Audio File to Object storage
    name = 'offensive_language.wav'
    audio_to_object_storage(main_bucket_name, namespace, config, name, file_location_local_audio) #passing fixed main bucket name and namespace. These are in main.py defined
    
    # Run Speech Model - Returns Results object
    object_name = run_speech_model(main_bucket_name, namespace, compartment_id_input, config, name)
    
    # Get Results File from Object Storage and Parse Transcription
    transcription_out = parse_results(main_bucket_name, namespace, config, object_name)
    transcription = transcription_out[0:498]  #filter on max 500 characters
    
    # Run Language Models on Transcription to Get Key Phrases and Sentiment
    key_phrases_string, sentiment_results = run_language_models(config, transcription_out)
    
    
    #convert list to string
    sentiment_result_string_output = 'input: '
    for sentimentx in sentiment_results:
        sentiment_result_string_output += sentimentx[0] + ', '
    
    sentiment_result_string = sentiment_result_string_output[0:498]  #filter on max 500 characters
        
    # Count Negative Aspects
    neg_aspects = 0 
    
    for sentiment in sentiment_results:
        if sentiment[1] == 'Negative':
            neg_aspects += 1

    
    return transcription, key_phrases_string, sentiment_result_string, neg_aspects
