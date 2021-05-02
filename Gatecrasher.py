#!/usr/bin/env python
# coding: utf-8

# %% Precursor

#!pip install -U pip 
#!pip install spotipy --upgrade
#!pip install -U scikit-learn
#!pip install MiniSom


# %% Import libraries

import json
import matplotlib.pyplot as pp
import numpy as np
import os
import pandas as pd
import pickle
import random
import requests
import sklearn
from sklearn import preprocessing
import urllib
print('Libraries imported')


# %% Import Spotipy

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
print('spotipy imported')


# %% Add Spotify credentials 

os.environ['SPOTIPY_CLIENT_ID']='b0efb30cca994b538e565fb9187fadc9'
os.environ['SPOTIPY_CLIENT_SECRET']='313904f3497f4f0cb194a99763083e6a'
os.environ['SPOTIPY_USERNAME']='kwanyeechan'
os.environ['SPOTIPY_REDIRECT_URI']='https://localhost:8000'


# %% Part 1 - Get Playlist Data 

# %% Add demo playlist data

SCOPE='user-library-modify playlist-modify-public'
REDIRECT_URI='https://localhost:8000'
PLAYLIST_ID = '3k8Cmr6OWfGqbEUvjAiTbS' 


# %% Spotify Authentication

print('##########')
print('##########')
print('##########')
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(),auth_manager=SpotifyOAuth(scope=SCOPE))
print('Authentication occured')


# %% Get a Playlist's Items

max_tracks = 100 
offset = 0

content = spotify.playlist_tracks(playlist_id=PLAYLIST_ID, 
                                  fields=None, 
                                  limit=max_tracks, 
                                  offset=offset, 
                                  market=None)

# %%  Compile Track Ids into DataFrame

track_ids = []
track_names = []
artist_ids = []
artist_names = []
added_by_ids = []
track_no = 0

while track_no < max_tracks:
    try: 
        track_id = content['items'][track_no]['track']['id']
        track_ids.append(track_id)
        track_name = content['items'][track_no]['track']['name']
        track_names.append(track_name)
        artist_id = content['items'][track_no]['track']['artists'][0]['id']
        artist_ids.append(artist_id)
        artist_name = content['items'][track_no]['track']['artists'][0]['name']
        artist_names.append(artist_name)
        added_by_id = content['items'][track_no]['added_by']['id']
        added_by_ids.append(added_by_id)
        track_no +=1
    except IndexError:
        break

# %%

df_track_id = pd.DataFrame(track_ids, columns=['track_id'])
df_track_name = pd.DataFrame(track_names, columns=['track_names'])
df_artist_id = pd.DataFrame(artist_ids, columns=['artist_ids'])
df_artist_name = pd.DataFrame(artist_names, columns=['artist_names'])
df_added_by_id = pd.DataFrame(added_by_ids, columns=['added_by_ids'])

df_track_properties = pd.concat([df_track_id, df_track_name, df_artist_id, df_artist_name, df_added_by_id], axis=1)


# %% Get Audio Features for Tracks and compile data into DataFrame

features = spotify.audio_features(track_ids[0])

# Features 'key', 'mode' are not considered

index = 0
features_list = []

for track_id in track_ids:
    features = spotify.audio_features(track_id)
    features_list.append([
                          features[0]['danceability'],
                          features[0]['energy'],
                          features[0]['loudness'],
                          features[0]['speechiness'],
                          features[0]['acousticness'],
                          features[0]['liveness'],
                          features[0]['valence'],
                          features[0]['tempo'],
                          features[0]['duration_ms'],
                          features[0]['time_signature']
                        ])

df_audio_features = pd.DataFrame(features_list, columns=['danceability', 
                                                         'energy',
                                                         'loudness',
                                                         'speechiness',
                                                         'acousticness',
                                                         'liveness',
                                                         'valence',
                                                         'tempo',
                                                         'duration_ms',
                                                         'time_signature'])

df_track_audio_features = pd.concat([df_track_properties, df_audio_features], axis=1)


# %% Calculate playlist statistics for ANN SOM 


# %% Part 2 - Get User Data 

# %% Get User's Top (Artists and) Tracks

# returned_user_top_tracks = spotify.current_user_top_tracks(limit=20, offset=0, time_range='medium_term')

# %% Get List of a User's Playlists


# %% Get Playlist's Items


# %% Get Audio Features


# %% Part 2 - Get Spotify recommendations

returned_genre_seeds = spotify.recommendation_genre_seeds()
short_genre_seeds_list = [returned_genre_seeds['genres'][86], returned_genre_seeds['genres'][99]]
seed_genre = random.choice(short_genre_seeds_list)
print(seed_genre)


# %%

no_gatecrasher_tracks = 20
random_df_row = df_track_audio_features.sample(n=1)
seed_artist = random_df_row.iloc[:,2].values
seed_track = random_df_row.iloc[:,0].values
print(seed_artist)
print(seed_track)


# %%

returned_gatecrasher_tracks = spotify.recommendations(seed_artists=seed_artist, seed_genres=seed_genre, seed_tracks=seed_track, limit=no_gatecrasher_tracks)
#                                                       country=None,
#                                                       min_acousticness=0, max_acousticness=1, target_acousticness=None,
#                                                       min_danceability=0, max_danceability=1, target_danceability=None,
#                                                       min_duration_ms=None, max_duration_ms=None, target_duration_ms=None,
#                                                       min_energy = 0, max_energy= 1, target_energy=None,
#                                                       min_instrumentalness=0, max_instrumentalness=1, target_instrumentalness=None,
#                                                       min_liveness=0, max_liveness=1, target_liveness=None,
#                                                       min_loudness=-10, max_loudness=1, target_loudness=None,
#                                                       min_speechiness=0, max_speechiness=0.2, target_speechiness=None,
#                                                       min_tempo=None, max_tempo=None, target_tempo=None,
#                                                       min_time_signature=None, max_time_signature=None, target_time_signature=None,
#                                                       min_valence=0, max_valence=1, target_valence=None)

gatecrasher_track_ids = [returned_gatecrasher_tracks['tracks'][0]['id']]

# %% Add gatecrasher track to playlist

if gatecrasher_track_ids not in df_track_audio_features.iloc[:,0].values:
    add_to_playlist = spotify.playlist_add_items(playlist_id=PLAYLIST_ID, 
                                                 items=gatecrasher_track_ids, 
                                                 position=None)
    print('added tracks to playlist')
else:
    print('track already on playlist')    


# %% Part 3 - Machine Learning 

# Use ANN SOM to find imposter tracks

# %% Import dataset

# dataset = df_track_audio_features
# X = dataset.iloc[:,5:-1].values
# y = dataset.iloc[:,-1].values
# print(X)
# print(y)


# %% Feature scaling

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range=(0,1))
#X_ft = sc.fit_transform(X)
#print(X_ft)


# %% Import SOM and train

# from minisom import MiniSom
# som = MiniSom(10,10,input_len=11,sigma=1.0,learning_rate=0.5)
# som.random_weights_init(X)
# som.train_random(data = X, num_iteration = 100)
# print('SOM trained')


# %% Finding winners or imposters

# mappings = som.win_map(X)
# print(mappings)


# %% Visualise the SOM 

# for i, x in enumerate(X):
#     w = som.winner(x)
#     print(w)
#     print(y[i])


# %%

# from pylab import cool, pcolor, colorbar, plot, show
# cool()
# pcolor(som.distance_map().T)
# colorbar()
# markers = ['o', 's']
# colors  = ['r', 'g']
# for i, x in enumerate(X):
#     w = som.winner(x)
#     plot(w[0] + 0.5,
#          w[1] + 0.5,
#          markers[y[i]],
#          markeredgecolor = colors[y[i]],
#          markerfacecolor = 'None',
#          markersize = 10,
#          markeredgewidth = 2)
# show()


# %% Export SOM and load it again 

# with open('som.p', 'wb') as outfile:
#     pickle.dump(som, outfile)


# with open('som.p', 'wb') as outfile:
#     pickle.dump(som, outfile)


# %% Is a new added track a gatecrasher?



