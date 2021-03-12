# -*- coding: utf-8 -*-

# Sample Python code for youtube.videos.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os

from decouple import config

import googleapiclient.discovery
import googleapiclient.errors

scopes = ["https://www.googleapis.com/auth/youtube"]

def get_video_metadata_by_id(youtube_video_id: str = None):
    if youtube_video_id is None:
        return

    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    # Get credentials and create an API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=config('YOUTUBE_API_KEY'))

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=youtube_video_id,
        maxResults=1
    )
    
    try:
        return request.execute()
    except googleapiclient.errors.HttpError:
        return None

