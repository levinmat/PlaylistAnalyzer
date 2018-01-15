#!/usr/local/bin/python

"""
TODO: 
   unittest
   test CL args - all
   sample_output - graphs, benchmarking classify
   benchmark function outside of mlp_classify
   comments - for each function, top of file, inner code
   README.md

"""


"""

Matt Levin - 2018

Program to play around with Spotify Python2.7 wrapper Spotipy (https://github.com/plamere/spotipy) specifically
to analyze a singular playlist (designed for Top 100 playlists) or compare multiple playlists.

After seeing my own and other's Top 100 of 2017 playlists, I wanted to play around with the data. The
first function I implemented counts the highest occurring artists in the playlist, including if they are not
the track's primary artist. Next I wanted to explore the audio features of the tracks, so I wrote a function
to extract the audio features and write them, and track metadata, for a playlist to a CSV file. This CSV
is then used for various analysis tasks.

I then pivoted towards using not just my own, but also my friends' Top 100 playlists. This opened more
possibilities, such as creating a "Dance Party" playlist from the top N most danceable songs from each 
playlist. Similarly, creating a "Study Party" playlist of the top N most acoustic songs. Currently you can
graph features against each other for any subset of the playlists. In the future, I'm going to work on
a classifier to identify whether the user will like an unlabeled song based on it's audio features and the
audio features of the tracks in their Top 100 playlist, as well as implement clustering to try to find 
interesting insights. 


Note: The Application credentials, my Spotify username, and a map of playlist names to the Spotify ID's is 
in my_info.py, which is hidden from GitHub. Therefore this code will not run if forked/copied from Git,
however example output can be found in the sample_output folder.
      
~ I have no affiliation with Spotify ~
      
"""

import spotipy  # Spotify Python wrapper (https://github.com/plamere/spotipy)
import spotipy.util as util

import sys
import os       # For environment variables (application credentials)
import csv      # For writing CSV file for each playlist
import glob
import argparse
from datetime import datetime
import heapq    # Heap data structure
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt # For plotting
# Clustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
# Classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB



"""
Writes a CSV file for the passed playlist that contains the metadata and audio features for each 
track in the playlist.
PARAMS:
     playlist - Spotify playlist object (full) to be analyzed
RETURNS:
     None
"""
def write_csv(playlist):

    outfile = "csv/" + playlist['name'].replace(' ', '_').lower() + ".csv"
    
    print("\nWriting playlist's tracks metadata and audio feature analysis to {}...".format(outfile))
    metadata = {} # Metadata = {track_id: {'name': Track Name, 'artist': Primary Artist, 'album': Album}}
    
    # Extract the metadata (ID, Name, Primary Artist, and Album) from each track
    for item in playlist['tracks']['items']:
        track = item['track']
        
        track_id = track['id'].encode('utf-8')                
        song_name = track['name'].encode('utf-8')
        primary_artist = track['artists'][0]['name'].encode('utf-8') # Just the primary/first artist
        album = track['album']['name'].encode('utf-8')

        metadata[track_id] = {'name': song_name, 'artist': primary_artist, 'album': album}
    
    # Query Spotify API to get the audio feature analysis for the tracks 
    all_tracks_features = sp.audio_features(metadata.keys()[:50]) # This work's fine if there's <50 tracks too
    all_tracks_features += sp.audio_features(metadata.keys()[50:]) # Limit for audio_features endpoint is 50
    
    # Open the CSV (rewrites old data, creates if doesn't exist)
    writer = csv.writer(open(outfile, 'w+')) 
    
    header_row = ["Track Name", "Primary Artist", "Album"] # The header row of the CSV, first the Metadata
    for feature in all_tracks_features[0].keys(): # Add all the audio feature names to the header row
        if feature not in ['track_href', 'analysis_url', 'uri', 'type']: # Ignore some columns
            header_row.append(feature.encode('utf-8')) 
    writer.writerow(header_row) # Write header row

    # Now go through the features for each track
    for i in range(len(all_tracks_features)):
        #print("Processing song: {}".format(i))
        track_features = all_tracks_features[i] # The features for this track
        if track_features is None:
            break # For playlists shorter than 100 tracks
        track_metadata = metadata[track_features['id']]
        # Start the row as the metadata for the track
        row = [track_metadata['name'], track_metadata['artist'], track_metadata['album']]
        # Then add each audio feature
        for (feature, feature_value) in track_features.items():
            if feature not in ['track_href', 'analysis_url', 'uri', 'type']: # Ignore some columns
                row.append(feature_value)
        writer.writerow(row) # Write the row to the CSV


"""
Writes a CSV for every playlist that has "Top 100" in the name for the user (me).
I gathered a bunch of Top 100 playlists from friends and this way I can just write all the CSV's in
one function call. 
PARAMS:
     None
RETURNS:
     None
"""
def write_all_csvs():
    all_playlists = sp.current_user_playlists(limit=50) # Returns simplified playlist objects
    selected_playlists = []
    for playlist in all_playlists['items']:
        if 'Top 100' in playlist['name']:
            selected_playlists.append(playlist)
    for p in selected_playlists:
        p_uri = p['uri'].split(':')[-1] # Get the ID of the playlist
        playlist = sp.user_playlist(username, p_uri) # Get the full playlist object
        write_csv(playlist)

"""
Analysis class contains several functions (with more coming) to perform analysis on one or more Top 100
playlists to try to extract some interesting insights. Decided to use a class as to not have to re-read 
the csv(s) each time they were needed. 

Also provides functions to create a Dance Party or Study Party playlist from input playlists. 

"""
class Analysis:
    """
    Initializer reads CSV files into self.data[file] for later use
    PARAMS:
         self
         files [Optional] - Array of filenames, which files to use, defaults to 'all' which uses all the files 
    RETURNS:
         None
    """
    def __init__(self, files='all'):
        if(files == 'all'): # Defaults to all files in the csv folder
            files = glob.glob('csv/*')
        
        self.data = {} # Map: { playlist_name : pandas_dataframe }
        # The playlist names are lower case and replace ' ' with '_' (done in make_csv)
        for file in files:
            # Strip the 'csv/' and the '.csv' and add it's Pandas dataframe to the data map
            self.data[file.split('/')[1].split('.')[0]] = pd.read_csv(file, skipinitialspace=True)
            
        self.all_data = pd.concat(self.data.values(), keys=self.data.keys()) # All data in one dataframe
        self.normalize_columns() # Creates tempo_0_1 and loudness_0_1 0-1 scaled columns from tempo and loudness
        
    """
    Normalize the quantatative columns in self.all_data that are not on a 0 to 1 scale 
    already (tempo and loudness)
    PARAMS:
         self
    RETURN:
         None
    """
    def normalize_columns(self):
        # Just min/max based 0-1 scaling, ignores original scale/units
        tempo = self.all_data['tempo']
        self.all_data['tempo_0_1'] = (tempo - tempo.min()) / (tempo.max() - tempo.min())
        
        loudness = self.all_data['loudness']
        self.all_data['loudness_0_1'] = (loudness - loudness.min()) / (loudness.max() - loudness.min())
    
    """
    Creates a playlist consisting of the N most danceable songs from each playlist passed to the function.
    First combines danceability and energy features then picks to top N from each playlist.
    PARAMS:
         self
         playlists [Optional] - The array of playlist CSV filenames to use, defaults to 'all' to use all
         n [Optional] - The number of tracks to use from each playlist, defaults to 3
    RETURNS:
         None (Playlist is created in Spotify with the name "Dance Party! [Today's Date]")
    """
    def dance_party(self, playlists='all', n=5):
        # For debugging... If R, A, and Ri were to have a dance party
        playlists = ['p_top_100', 'a_top_100', 'ri_top_100', 'm_top_100']
        if(playlists == 'all'):
            playlists = self.data.keys()        
        
        tracks = []
        # TODO: Maybe add some randomization so it's not always the same for the same input playlists?
        for playlist in playlists:
            df = self.data[playlist]
            df['dance_value'] = df['danceability'] + df['energy'] # New column for the combination
            df = df.sort_values('dance_value', ascending=False) # Sort by new column
            tracks += df.head(n)['id'].values.tolist()
        tracks = list(set(tracks)) # Remove duplicates
        np.random.shuffle(tracks) # Shuffle playlist order
        # TODO: Temporal analysis to try to make tracks flow into each other better
        p_name = 'Dance Party! ' + datetime.now().strftime('%m/%d/%y') # Playlist name
        sp.user_playlist_create(username, name=p_name, public=False) # Create playlist
        new_playlist = find_playlist(p_name) # Find the newly created playlist (helper function)        
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks) # Add the tracks to playlist
        print("'{}' playlist created successfully!".format(p_name))
      
      
    """
    Creates a playlist consisting of the N songs from each playlist that would be best for a study session.
    Uses several audio features: instrumentalness, acousticness, energy, speechiness
    PARAMS:
         self
         playlists [Optional] - The array of playlist CSV filenames to use, defaults to 'all' to use all
         n [Optional] - The number of tracks to use from each playlist, defaults to 5
    RETURNS:
         None (Playlist is created in Spotify with the name "Study Party! [Today's Date]")
    """
    def study_party(self, playlists='all', n=5):
        # For debugging... If L, Am, Z, and J were to have a study party
        playlists = ['l_top_100', 'am_top_100', 'z_top_100', 'j_top_100']
        if(playlists == 'all'):
            playlists = self.data.keys()
            
        tracks = []
        # TODO: Maybe add some randomization so it's not always the same for the same input playlists?        
        for playlist in playlists:
            df = self.data[playlist]
            df['study_value'] = (2 * df['instrumentalness']) + df['acousticness'] +  \
                                (2 * (1 - df['energy'])) + (1 - df['speechiness'])
            df = df.sort_values('study_value', ascending=False)
            tracks += df.head(n)['id'].values.tolist()
        tracks = list(set(tracks)) # Remove duplicates
        np.random.shuffle(tracks) # Shuffle playlist order
        # TODO: Temporal analysis to try to make tracks flow into each other better       
        p_name = 'Study Party! ' + datetime.now().strftime('%m/%d/%y') # Playlist name
        sp.user_playlist_create(username, name=p_name, public=False) # Create playlist
        new_playlist = find_playlist(p_name) # Find the newly created playlist (helper function)        
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks) # Add the tracks to playlist
        print("'{}' playlist created successfully!".format(p_name))
        
        
    """
    Plots the clusters (in different colors)
         - First a bar chart showing each playlists' distributions of clusters
         - Then, for each combination of features (2D), shows all playlists' data combined on a scatter plot
            with each cluster in a different color. 
    PARAMS:
        self
        n_clusters - The number of distinct clusters used
        algorithm - {'GMM' or 'spectral'} The algorithm you want to plot (which column of all_data to use) 
        x [Optional] - The feature to use for the x axis, defaults to None which cycles through all features
        y [Optional] - The feature to use for the y axis, defaults to None which cycles through all features
        features [Optional] - The list of features to plot, each used as the x and y axis with all others
    RETURNS:
        Nothing, displays plots in a new window (one at a time)
    
    """
    def plot_clusters(self, n_clusters, algorithm='GMM', 
                      x=None, y=None,
                      features=['energy', 'liveness', 'tempo_0_1', 'speechiness', 
                                'acousticness', 'instrumentalness', 'danceability', 
                                'loudness_0_1', 'valence']):
        
        
        # Bar chart - Cluster count for each file (playlist)
        fig, ax = plt.subplots() 
        counts = np.zeros(shape=(len(self.data.keys()), n_clusters))
        file_names = self.data.keys()
        for f in range(len(file_names)):
            data = self.all_data.loc[file_names[f]]
            counts[f] = [0 for _ in range(n_clusters)]
            for cluster, group in data.groupby([algorithm]):
                counts[f][cluster] = len(group)
        counts_df = pd.DataFrame(counts)
        counts_df.plot.bar()
        
        # Scatterplot of clusters (each pair of features used)
        for x_feature in features:
            # If x or y is set, skip all pairs except the desired x and/or y values
            if x != None and x_feature != x:
                continue
            for y_feature in features:
                if y != None and y_feature != y:
                    continue
                if x_feature != y_feature:
                    fig, ax = plt.subplots()
                    colors = iter(plt.cm.rainbow(np.linspace(0,1,n_clusters)))
                    for name, group in self.all_data.groupby([algorithm]):
                        ax = group.plot.scatter(ax=ax, x=x_feature, y=y_feature,
                                                alpha=0.7, label=name, c=next(colors))
                    plt.xlabel(x_feature.title())
                    plt.ylabel(y_feature.title())
                    plt.title(y_feature.title() + ' versus ' + x_feature.title())
                    plt.show()               

            
    """
    Performs Gaussian mixture model clustering on the data. Varies the number of clusters 
    in order to find the best fitting clustering.
    
    PARAMS:
        show_plot [Optional] - Boolean, whether or not to call plot_clusters after, defaults to True
        features [Optional] - The features to use in the clustering
    RETURNS:
        Nothing, sets the 'GMM' column in all_data to the cluster indices from the optimal clustering
    
    """         
    def gmm_cluster(self, show_plot=True,
                    features=['energy', 'liveness', 'tempo_0_1', 'speechiness', 
                                    'acousticness', 'instrumentalness', 'danceability', 
                                    'loudness_0_1', 'valence']):

        X = self.all_data[features]        
        best_score = None  
        best_n = None
        for n in range(1,15): # Find the optimal number of components
            gm = GaussianMixture(n_components=n, covariance_type='full', max_iter=300, n_init=5)
            gm.fit(X)
            score = gm.bic(X)
            if best_score == None or score < best_score:
                best_score = score
                best_n = n
                best_gm = gm
        print('Best Score: {} with N = {}'.format(best_score, best_n))
        
        Y = best_gm.predict(X)
        self.all_data['GMM'] = Y
        
        if show_plot:
            self.plot_clusters(best_n, features=features, algorithm='GMM')


    """
    Performs spectral clustering on all the data, using n clusters.

    PARAMS:
         show_plot [Optional] - Boolean, whether or not to call plot_clusters after, defaults to True
         n_clusters [Optional] - The number of clusters to use, defaults to 5
         features [Optional] - The features to use in the clustering
    RETURNS:
         Nothing, sets the 'spectral' column in all_data to the predicted clusters
    """
    def spectral_cluster(self, show_plot=True, n_clusters=5,
                         features=['energy', 'liveness', 'speechiness', 
                                   'acousticness', 'instrumentalness', 'danceability', 
                                    'valence', 'loudness_0_1', 'tempo_0_1']):

        X = self.all_data[features]
        spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                      affinity="nearest_neighbors", n_init=5)
        Y = spectral.fit_predict(X)
        self.all_data['spectral'] = Y

        if show_plot:
            self.plot_clusters(n, features=features, algorithm='spectral')
            
            

    """
    Uses a Multilayer Percepetron classifier to assign a playlist to each song based on its audio features.
    Varies the sizes/number of hidden layers in order to find the best performing model, using multiple
    seeds to optimize the solution found. Uses all data for training, and the training score to find the best
    model (as opposed to seperating testing data) due to the amount/nature of the data.
    
    PARAMS:
        features [Options] - Features to use in the classification 
    
    
    """
    def classify_mlp(self, features=['energy', 'liveness', 'speechiness', 
                                      'acousticness', 'instrumentalness', 'danceability', 
                                    'valence', 'loudness_0_1', 'tempo_0_1']):

        # Make sure any included generated columns have been created before starting
        # i.e. if GMM clusters are to be used in classification but have not been generated yet
        self.check_features(features) 

        # Create X and Y matrices
        X, Y = [], []
        for file_name in self.data.keys():
            X_i = self.all_data.loc[file_name]
            for x in X_i[features].values:
                X.append(x)
                Y.append(file_name)

        best_score = (0,(0,),0) # Best trial: (score, layers, seed)
        layer_options = [(150,200,100),(200,500,50),(150,250,350,200,100)]
        seeds = [0,101,2002,30003,400004,5000005,60000006,700000007,123456789,999999999]
        layer_scores = [[] for _ in layer_options]
        for i in range(len(layer_options)):
            layers = layer_options[i]
            print('\nLayers: {}'.format(layers))
            for seed in seeds:
                # Create MLP classifier, train, count how many were classified correctly
                mlp = MLPClassifier(hidden_layer_sizes=layers,
                                   activation="relu", 
                                   solver='adam', 
                                   alpha=0.0001, 
                                   learning_rate="constant", 
                                   learning_rate_init=0.001, 
                                   max_iter=1000, 
                                   shuffle=True, 
                                   random_state=seed, 
                                   tol=1e-4)

                mlp.fit(X,Y)
                predict_Y = mlp.predict(X)
                score = np.sum(Y == predict_Y) # Score is number of correct predictions
                print('{} classified correctly out of {} for seed {}.'.
                      format(score, len(Y), seed))
                if score > best_score[0]:
                    best_score = (score, layers, seed)
                layer_scores[i].append(score)    
        
        
        # Print results
        print('\n')
        for i in range(len(layer_options)):
            print('Average score: {:.1f} with SD: {:.2f} and Min: {} for layers = {}'.
                  format(np.average(layer_scores[i]), np.std(layer_scores[i]), 
                         np.min(layer_scores[i]), layer_options[i]))
        
        print('\nThe best trial scored {} with layers {} and seed {}'.format(*best_score))
        print('Seeds Used = {}\nFeatures Used = {}'.format(seeds, features))
        
        # Set self.mlp_classifier to best performing model for later use
        self.mlp_classifier = MLPClassifier(hidden_layer_sizes=best_score[1],
                                   activation="relu", 
                                   solver='adam', 
                                   alpha=0.0001, 
                                   learning_rate="constant", 
                                   learning_rate_init=0.001, 
                                   max_iter=1000, 
                                   shuffle=True, 
                                   tol=1e-4)
        self.mlp_classifier.fit(X,Y)
    
    
    def mlp_predict_playlist(self, playlist='ri_top_100', n=100, 
                             features=['energy', 'liveness', 'speechiness', 
                                      'acousticness', 'instrumentalness', 'danceability', 
                                      'valence', 'loudness_0_1', 'tempo_0_1']):
        
        # Create X and Y matrices
        X, Y, song_ids = [], [], []
        for file_name in self.data.keys():
            X_i = self.all_data.loc[file_name]  
            for song_id in X_i[['id']].values:
                song_ids.append(song_id[0])            
            for x in X_i[features].values:
                X.append(x)
                Y.append(file_name)
        
        try:
            predict_confidence = self.mlp_classifier.predict_proba(X)
        except AttributeError: # If haven't found the best classifier yet, do so first
            #self.classify_mlp(features) 
            # To save time, I just hardcoded the best performing layer sizes so far: (150,200,100)
            self.mlp_classifier = MLPClassifier(hidden_layer_sizes=(150,200,100),
                                   activation="relu", 
                                   solver='adam', 
                                   alpha=0.0001, 
                                   learning_rate="constant", 
                                   learning_rate_init=0.001, 
                                   max_iter=1000, 
                                   shuffle=True, 
                                   tol=1e-4)
            self.mlp_classifier.fit(X, Y)
            predict_confidence = self.mlp_classifier.predict_proba(X)
        
        # Figure out which column relates to the selected playlist
        index = np.where(self.mlp_classifier.classes_==playlist)[0][0]
        # The liklihood that each song is from the selected playlist
        scores = predict_confidence[:,index]
        # Pick the most likely n songs
        score_heap = []
        for i in range(len(scores)):
            heapq.heappush(score_heap, (scores[i], song_ids[i]))
        selected_songs = heapq.nlargest(n, score_heap)
        
        tracks = []
        for _,track_id in selected_songs:
            tracks.append(track_id)
        tracks = list(set(tracks)) # Remove duplicates
        np.random.shuffle(tracks)
        
        p_name = playlist.split('_')[0].title() + ' Generated'
        sp.user_playlist_create(username, name=p_name, public=False) # Create playlist
        new_playlist = find_playlist(p_name) # Find the newly created playlist (helper function)        
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks) # Add the tracks to playlist                   
        print('{} playlist created successfully! Check Spotify'.format(p_name))
        
        
        
    def classify_naive_bayes(self, features=['energy', 'liveness', 'speechiness', 
                                            'acousticness', 'instrumentalness', 'danceability', 
                                            'valence', 'loudness_0_1', 'tempo_0_1']):
        

        # Make sure any included generated columns have been created before starting
        self.check_features(features)
        
        # Create X and Y matrices
        X, Y = [], []
        for file_name in self.data.keys():
            X_i = self.all_data.loc[file_name]            
            for x in X_i[features].values:
                X.append(x)
                Y.append(file_name) 
        
        # Create Naive Bayes, train, predict from X and see how many are correctly matched        
        nb = GaussianNB()
        nb.fit(X,Y)
        predict_Y = nb.predict(X)
        print('{} classified correctly out of {}.'.format(np.sum(Y == predict_Y), len(Y)))
     
     
    def check_features(self, features):
        for f in features:
            if f not in self.all_data.keys():
                features.remove(f)
                if f == 'GMM':
                    self.gmm_cluster(show_plot=False, features=features)
                elif f == 'spectral':
                    self.spectral_cluster(show_plot=False, features=features)
                else:
                    print('Unrecognized feature: {}'.format(f))
                    sys.exit(0)
    
# END OF ANALYSIS CLASS #
    
"""
Finds the artists who occur the most frequently in the given playlist. Takes into account the fact that
some songs have multiple artists. Prints the top N occurring artists and how many tracks of theirs are in
the playlist. 
PARAMS:
     playlist - The Spotify playlist object (full)
     n [Optional] - The number of artists to display, defaults to -1 to print all artists present
RETURNS:
     None, just prints results to stdout
"""
def top_artists(playlist, n=-1):
    
    artists_map = {} # The songs each artist has (takes into account that some songs have >1 artist)
    # Indexed by artist name - {artist_name: [song_names]}

    # For each track in the playlist
    for track in playlist['tracks']['items']:        
        track_name = track['track']['name'].encode('utf-8')
        # For each artist listed for the track
        for artist in track['track']['artists']:
            artist_name = artist['name'].encode('utf-8')
            # Add the track to that artist's list of songs
            if artist_name not in artists_map:
                artists_map[artist_name] = [track_name]
            else:
                artists_map[artist_name].append(track_name)
    
    # Create the heap to find the top occurring artists
    heap = []
    for (artist, songs) in artists_map.items():
        #print("{} has {} songs: {}".format(artist, len(songs), songs))
        heapq.heappush(heap, (len(songs), artist))
    
    # If n was not set (defaults to -1) then print all artists
    if(n == -1):
        n = len(artists_map.keys())
        print('Top artists in playlist:')
    else:
        print('Top {} artists in playlist:'.format(n))    
    # Print them in order in a column format (I wish CCR picked a shorter name!)
    for (num, artist) in heapq.nlargest(n, heap):
        print("{1:<8} -  {0:<30}".format(artist[:30],  # Only first 30 characters of song name
                                           str(num) + (" song " if num==1 else " songs")))  # Plural if >1
    
    # Diversity (can be >100 because some songs have multiple artists)
    print("\n{} different artists appeared in this playlist.".format(len(artists_map.keys())))


# Helper function, finds the playlist with the given name for the current user
def find_playlist(name):
    playlists = sp.user_playlists(username)
    for p in playlists['items']:
        if p['name'] == name:
            return p
    return None



#============================================================================
class TestAnalyzer(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(self):
        """ runs once before ALL tests """
        print("\n........... starting unit testing ..................")
        self.analyzer = Analysis()

    def setUp(self):
        """ runs once before EACH test """
        pass
    
    def test_top_artists(self):
        playlist_id = '38hwZEL0H1z6wUbq0UoHBS'
        playlist = sp.user_playlist(username, playlist_id)
        
    def test_predict_mlp(self):
        self.analyzer.mlp_predict_playlist()
        
        
#============================================================================


"""
Main Method, parses command line arguments and calls the appropriate function.
Will eventually accept more CL arguments as I expand other functionalities and generalize the code further
(Such as playlist ID, username, which graphs to create, which features to use, etc.)

"""
if __name__ == '__main__':
    
    # Access enviornment variables for authenticating with Spotify
    try:
        username = os.environ['SPOTIFY_USERNAME']
        client_id = os.environ['SPOTIFY_CLIENT_ID']
        client_secret = os.environ['SPOTIFY_CLIENT_SECRET']
        redirect_uri = os.environ['SPOTIFY_REDIRECT_URI']
    except KeyError:
        print('The following environment variables must be set in order to run this script:\n' +
              'SPOTIFY_USERNAME, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI')    
        sys.exit(0)
        
    # Request access token from Spotify
    scope = 'playlist-read-collaborative playlist-read-private playlist-modify-private'
    token = util.prompt_for_user_token(username, scope, 
                                       client_id=client_id, 
                                       client_secret=client_secret, 
                                       redirect_uri=redirect_uri)   
    
    if token is None: # Unable to authenticate with Spotify
        print("Unable to get access token.")  
        sys.exit(0)
    
    # If successfully created an authentication token with Spotify 
    sp = spotipy.Spotify(auth=token)
    analyzer = Analysis()
    
    if(len(sys.argv) > 1): # Parse command line arguments
        feature = sys.argv[1] # Which feature is being used
        for i in range(2, len(sys.argv), 2): # Parse any optional arguments into args dict
            args[sys.argv[i]] = sys.argv[i+1]
        
        # Top Artists
        if feature == '-a':
            n = 15 # Default to showing top 15 artists
            if '-n' in args:
                n = int(args['-n'])
            playlist_id = '38hwZEL0H1z6wUbq0UoHBS' # Default playlist to analyze
            if '-p' in args:
                playlist_id = args['-p']
            playlist = sp.user_playlist(username, playlist_id)
            top_artists(playlist, n)
        
        # Write CSV of audio features for given playlist (or all user's Top 100 playlists if none provided)
        elif feature == '-w':
            if '-p' in args: # Of a single playlist if -p argument provided
                playlist_id = args['-p']
                playlist = sp.user_playlist(username, playlist_id)
                write_csv(playlist)
            else: # Default to all user playlists (that have 'Top 100' in the name)
                write_all_csvs()
        
        # Dance Party
        elif feature == '-d':
            # Usage: python playlist_analyzer.py -d -p PNAME_0,PNAME_1,... -n N
            # -p : Comma seperated lists of playlist CSV names (Lower Case, '_' instead of space)
            # -n : Number of songs to take from each playlist
            playlists = 'all'
            if '-p' in args:
                playlists = args['-p'].split(',') # Split on commas
                playlists = map(lambda s: s.split('.')[0], playlists) # Remove '.csv' if given
            n = 5
            if '-n' in args:
                n = int(args['-n'])
            analyzer.dance_party(playlists, n)
        
        # Study Party
        elif feature == '-s':
            playlists = 'all'
            if '-p' in args:
                playlists = args['-p'].split(',') # Split on commas
                playlists = map(lambda s: s.split('.')[0], playlists) # Remove '.csv' if given
            n = 5
            if '-n' in args:
                n = int(args['-n'])
            analyzer.study_party(playlists, n)
        
        # Show help message
        elif feature == '-h' or feature == '--help':
            print("""
This project contains several functions, use one of the following choices and its [optional] parameters:\n
-h or --help or -v\n\tShows this help message\n
-a [-p Playlist_ID] [-n Number_Of_Artists_To_Display]
\tShows top Artists in a Playlist (defaults to showing all top artists in order)\n
-w [-p Playlist_ID]
\tWrite CSV file of audio features for playlist (defaults to all Top 100 playlists)\n
-d [-n Number_Songs_From_Each_Playlist] [-p Playlist_CSV_Names_Seperated_By_Commas]
\tGenerates a Dance Party playlist from given Playlist CSV file names, taking n songs from each\n
-s [-n Number_Songs_From_Each_Playlist] [-p Playlist_CSV_Names_Seperated_By_Commas]
\tGenerates a Study Party playlist from given Playlist CSV file names, taking n songs from each\n""")

        # Invalid feature
        else:
            print("Invalid selection, please use '-h' or '--help' to show correct usage.")
   
    else: # Run unit testing when no arguments are given
        unittest.main()

      
    print('\nProgram Complete')
