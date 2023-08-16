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


##########################################################################################################################################
##########################################################################################################################################
######################################################## Use case 1          #############################################################
##########################################################################################################################################
##########################################################################################################################################

##########################################################################################################################################
######################################################## Function 1          #############################################################
##########################################################################################################################################

def input_youtube_video(YOUTUBE_URL):
    
    #delete previous videos
    os.system("rm -r /home/datascience/youtube_videos")

    #create a local directory to store the video
    path_input_locally = "/home/datascience/youtube_videos/" 

    try:       
        if not os.path.exists(path_input_locally):         
            os.makedirs(path_input_locally)    

    except OSError: 
        print ('Error: Creating directory for youtube video locally')
        

    #download file from youtube
    yt = YouTube(YOUTUBE_URL)

    #store in local folder
    stream = yt.streams.get_by_itag(22)
    file_name_random = str(uuid.uuid4())
    file_location_local = stream.download(output_path=path_input_locally, filename  = file_name_random + ".mp4")
    
    print("Youtube download completed and stored in " + str(file_location_local))
    
    return file_location_local

##########################################################################################################################################
######################################################## Function 2          #############################################################
##########################################################################################################################################

def input_profile_image(full_bucket_name):
    
    #create a local directory to store the image
    path_input_locally_image = "/home/datascience/profile_image/" 

    try:       
        if not os.path.exists(path_input_locally_image):         
            os.makedirs(path_input_locally_image)    

    except OSError: 
        print ('Error: Creating directory for profile image locally')

    print("Full bucket name is " + full_bucket_name)
    
    #get the image from the bucket and store locally
    fs = ocifs.OCIFileSystem()
    fs.invalidate_cache(full_bucket_name)
    fs.get((full_bucket_name + "*.jpg"), path_input_locally_image , recursive=True, refresh=True)
    
    #get file name    
    file = [os.path.basename(x) for x in glob.glob(path_input_locally_image + '*.jpg')]
    profile_image_name = file[0]
    profile_image_loc = path_input_locally_image + file[0]
    print("Profile image is stored locally at " + profile_image_loc)
        
    #delete input image from bucket to clear the bucket
    delete_object = os.path.join(full_bucket_name, file[0])
    fs.rm(delete_object, recursive=True)
    
    print("Image stored locally and removed from bucket")
    
    return profile_image_name, profile_image_loc


##########################################################################################################################################
######################################################## Predict             #############################################################
##########################################################################################################################################

def predict_video(full_bucket_name, YOUTUBE_URL, SCHEMA_NAME):
    
    # Delete images if there are images in the local folder already
    path_split_images = "/home/datascience/split_images"
    files = glob.glob('/home/datascience/split_images/*.jpg')

    for f in files:
        os.remove(f)

     #create a local folder to the images
    path_split_images = "/home/datascience/split_images"

    try:       
        # creating a folder named split_images 
        if not os.path.exists(path_split_images):         
            os.makedirs(path_split_images)    

    except OSError: 
        print ('Error: Creating directory of data for split images')
        
    ######
    ###### Function 1    
    file_location_local = input_youtube_video(YOUTUBE_URL)
    
    ######
    ###### Function 2
    profile_image_name, profile_image_loc = input_profile_image(full_bucket_name)  
    
    print("Fetching video from " + file_location_local)
    print("Fetching profile image from " + profile_image_loc)
    
    #####
    ##### Main function video_only
    
    # Read the video from specified path 
    cam = cv2.VideoCapture(file_location_local)
    
    #get fps of original video
    fps = cam.get(cv2.CAP_PROP_FPS)
    number_of_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    print("**************************************************************** Original fps in video is " + str(fps))
    
    # calculate duration of the video
    total_duration_video = round(number_of_frames / fps)
    
    #define list of frames to analyze. 
    list_of_frames = list(range(1, 9001, 30))  #max 5 minutes of video. #starts at frame 1, ends at frame 601 (which is 20 seconds at FPS = 30 and 10 seconds at FPS = 60) with 30 frames in between (= 1 second). So, takes 20 (= 21 seconds) frames from the video

    #loop through the video and cut into images
    currentframe = 0

    while(True):
        
        for frame in list_of_frames:
            cam.set(cv2.CAP_PROP_POS_FRAMES, frame)
            print("Analyze frame number " + str(frame))

            # reading from frame 
            ret,frame = cam.read()

            if ret:
                if currentframe < 10:   
                    name = path_split_images + '/frame000' + str(currentframe) + '.jpg'           

                elif currentframe >= 10 and currentframe < 100:   
                    name = path_split_images + '/frame00' + str(currentframe) + '.jpg'          

                elif currentframe >= 100 and currentframe < 1000:   
                    name = path_split_images + '/frame0' + str(currentframe) + '.jpg'   

                else:
                    name = path_split_images + '/frame' + str(currentframe) + '.jpg'      

                print ('Creating...' + name) 

                # writing the extracted images 
                cv2.imwrite(name, frame) 

                # increasing counter
                currentframe += 1
            
        else: 
            break

    cam.release()    
    
    #apply DeepFace to the images
    try:
        dfs = DeepFace.find(img_path = profile_image_loc, db_path = "/home/datascience/split_images", enforce_detection=False)  #first input is the profile image, second is the folder containing the split images
        
    except:
        pass #if no face is found in any of the images
    
    #get the dataframe of the results
    output_df = dfs[0]
    
    ########## calculations
    seconds_in_screen = output_df.shape[0]  # = total frames detected = frames per second as we are looping through each frame
    total_seconds_video_analyzed = total_duration_video
    
    #in screen calc
    list_in_screen = []
    if seconds_in_screen > 0:
        in_screen = 'Person was in video'
        print(in_screen)
        list_in_screen.append(in_screen)
    else:
        in_screen = 'Person was not in video'
        list_in_screen.append(in_screen)
        print(in_screen)
        
    output_in_screen = list_in_screen[0]
    
    print("**************************************************************** Person was or was not in screen is :" + str(output_in_screen))
    print("**************************************************************** Total seconds analyzed of entire video " + str(len(list_of_frames)) + " seconds")
    print("**************************************************************** This person was " + str(seconds_in_screen) + " seconds in screen")
    
    return output_in_screen, seconds_in_screen, total_seconds_video_analyzed
