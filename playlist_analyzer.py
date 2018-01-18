#!/usr/local/bin/python

"""
TODO: 
   sample_output - graphs, benchmarking classify
   comments - for each function, top of file, inner code
   README.md - Focus on MLP benchmarking, generating THEN dance/study lists THEN rand things: top artists, clustering, 
                Hype up the NN finding songs that I actually liked that weren't from my own

"""


"""

Matt Levin - 2018

This program contains several functions that explore analyzing playlists, designed for users'
Top 100 playlists, using the Spotify Python wrapper Spotipy (https://github.com/plamere/spotipy).

Please see README.md for instructions on using this script and more details on the features listed below.
The -h or --h flag can be used as well to show proper usage for each feature.

Features:
 - Train a classifier to predict which playlist a track belongs to based on its audio features using
   either a Multilayer Perceptron (MLP) classifier (Analyzer.classify_mlp) or a Naive Bayes classifier
   (Analyzer.classify_naive_bayes)
 - Use a trained MLP classifier to create a playlist of the tracks from all the Top 100 playlists that
   are deemed most likely to be from a given playlist, AKA songs the user would most likely enjoy based
   on their audio features (Analyzer.predict_playlist)
 - Create a playlist of the most danceable songs from each given source playlist (Analyzer.dance_party) or
   the songs that would be best to study to (Analyzer.study_buddies) for a group dance party or study session
 - Find the artists that have the most songs in a playlist, including if they're not the primary artist
   for the track (Analyzer.top_artists)
 - Extract the audio features and metadata for each track in a playlist and write into a CSV
   (Writer.write_csv) or for all the user's playlists with 'Top 100' in the name (Writer.write_all_csvs)
 - Perform Gaussian Mixture Model clustering (Analyzer.gmm_cluster) or Spectral 
   clustering (Analyzer.spectral_cluster) and plot the results (Analyzer.plot_clusters)
 - Perform benchmarking on the MLP classifier or the GMM clustering in order to find the model that
   works best for the data (Analyzer.benchmark_mlp or Analyzer.benchmark_gmm_cluster)


Code Outline:
 Analyzer class
   - Initializer
   - Classification functions (MLP and Naive Bayes) and Playlist generator using MLP (my favorite function)
   - Dance Party and Study Buddies playlist generator functions
   - Clustering functions (GMM and Spectral) and cluster plotting function
   - Top Artists function
   - Helper functions
 Writer class
   - Write a CSV for a playlist's metadata and audio features, or for each playlist conaining 'Top 100'
 TestAnalyzer class
   - Unit testing for all the different features
 Main Method
   - Authenticates with Spotify (requires environment varibles to be set)
   - Parses command line arguments to call appropriate function, or calls unittest.main() if no args given


Note: The following environment variables must be set in order to locally run the script:
      SPOTIFY_USERNAME, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
           
~ I have no affiliation with Spotify ~

"""

import spotipy  # Spotify Python wrapper (https://github.com/plamere/spotipy)
import spotipy.util as util

import sys
import os       # For environment variables (application credentials)
import csv      # For writing CSV file for each playlist
import glob
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
The Analyzer class contains most of the cool functions in this project. It's constructor reads the CSV
files specified (defaults to all CSV files in the csv folder) and it then is split into several sections
for the remaining functions:
- Classification: MLP and Naive Bayes classification, predicting a playlist using MLP, MLP benchmarking
- Clustering: GMM and Spectral clustering, and plotting the resulting clusters
- Dance Party and Study Buddies: Generate a dance or study playlist from the specified playlists
- Top Artists: Finds the highest occuring artists in a playlist
- Helper Functions

"""
class Analyzer:
    """
    Initializer reads CSV files for the given files (defaults to 'all' to read all present csv files) and
    creates a pandas dataframe (self.data) that contains the information for all, with an extra column 'source'
    which says which file (playlist) the track came from.
    
    PARAMS:
         files [Optional] - Array of filenames, which files to use, defaults to 'all' which uses all the files 
    RETURNS:
         None
    """
    def __init__(self, files='all'):
        if(files == 'all'): # Defaults to all files in the csv folder
            files = glob.glob('csv/*')
        
        # Read each selected csv file into it's own dataframe - Add a column 'source' with the filename
        dfs = [pd.read_csv(f).assign(source=os.path.basename(f).split('.')[0]) for f in files]
        
        # Combine the dataframes into self.data
        self.data = pd.concat(dfs, ignore_index=True)  
        # List of source playlists for easy access since it is needed a lot
        self.sources = self.data.source.unique()
        
        # Creates tempo_0_1 and loudness_0_1 0-1 scaled columns from tempo and loudness
        # (Uses min/max normalization as opposed to mean/deviation)
        tempo = self.data['tempo']
        self.data['tempo_0_1'] = (tempo - tempo.min()) / (tempo.max() - tempo.min())
    
        loudness = self.data['loudness']
        self.data['loudness_0_1'] = (loudness - loudness.min()) / (loudness.max() - loudness.min())


    #===============================================================================    
    # Start of Classification Functions
       
    """
    Trains a Multilayer Perceptron classifier to identify which source playlist each track came from
    based on its audio features. 
    
    PARAMS:
        layers   [Optional] - The hidden layer sizes used in the MLP
        seed     [Optional] - The random_state to use for training (None uses a random random_state)
        features [Optional] - Features to use in the classification 
    RETURNS:
        None, sets self.mlp_classifier to the trained model
    
    """
    def classify_mlp(self, layers=(150,200,100), seed=0,
                     features=['energy', 'liveness', 'speechiness', 
                                'acousticness', 'instrumentalness', 'danceability', 
                                'valence', 'loudness_0_1', 'tempo_0_1']):

        # Make sure any included generated columns have been created before starting
        # i.e. if GMM clusters are to be used in classification but have not been generated yet
        self.check_features(features) 

        # Create X and Y matrices
        X = self.data[features] # X is the selected features for each track
        Y = self.data[['source']].values.ravel() # Y is the source playlist each track came from

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

        mlp.fit(X,Y) # Fit to data
        self.mlp_classifier = mlp # Save for later use
      
    """
    Uses an MLPClassifier (neural network) to find the n songs which are most likely from the given playlist 
    and creates a Spotify playlist of them (shuffled order). When used with a Top 100 playlist, this creates a 
    playlist of songs the user would be most likely to enjoy (according to the MLP) consisting of songs
    from their own Top 100 playlist, and any others used in training the MLP. 
    
    Theoretically, the pool of songs chosen from can extend to more than just all the Top 100 playlists 
    combined, which would give even better results. Using a user's Top 200 or more songs would also lead 
    to better results, however from empirical testing, this works really well at creating a playlist that
    the user would enjoy.
    
    Note: Generated playlist may be less than n songs since duplicates can exist and are removed at the end.
    
    PARAMS:
        playlist [Optional] - The CSV file name (no .csv) of the Top 100 playlist of the seed user
        n        [Optional] - The number of songs to put in the generated playlist
        features [Optional] - The audio features to use in the neural network training and prediction
    RETURNS:
        None, creates a playlist in Spotify
    
    """
    def predict_playlist(self, playlist='my_top_100', n=100, 
                         features=['energy', 'liveness', 'speechiness', 
                                   'acousticness', 'instrumentalness', 'danceability', 
                                   'valence', 'loudness_0_1', 'tempo_0_1']):
        
        self.check_features(features) # Make sure features are valid and generated if applicable
        self.check_playlist(playlist) # Make sure playlist is a valid option (no '.csv' in filename)

        if not hasattr(self, 'mlp_classifier'): # If we haven't found the mlp_classifier yet...
            ##Find and set self.mlp_classifier to the best performing model using benchmark_mlp
            #self.benchmark_mlp(features=features)
            
            # To save time, I just hardcoded the best performing layer sizes I've found so far: (150,200,100)
            self.classify_mlp(layers=(150,200,100), seed=None, features=features)
            # Note: Since the random_state is not set here, it picks a different playlist each time (good!)           
        
        # Get the prediction probabilities for X (self.data[features])
        predict_confidence = self.mlp_classifier.predict_proba(self.data[features])
        # Figure out which column relates to the selected playlist
        playlist_index = self.mlp_classifier.classes_.tolist().index(playlist)
        # Isolate that column only as a 1D array
        scores = predict_confidence[:,playlist_index]
        
        # Pick the most likely n songs using a heap
        song_ids = self.data[['id']].values.ravel() # Song IDs as a 1D array        
        score_heap = [] # Heap used to order the Song IDs by their prediction confidence
        for score, song_id in zip(scores, song_ids):
            heapq.heappush(score_heap, (score, song_id))
        selected_songs = heapq.nlargest(n, score_heap) # Take the n most likely
        tracks = [track_id for _,track_id in selected_songs] # Isolate the IDs
        tracks = list(set(tracks)) # Remove duplicates
        np.random.shuffle(tracks)  # Shuffle order
        
        # Create the playlist in Spotify
        p_name = playlist.split('_')[0].title() + ' Generated' # Playlist name based off CSV name
        # (i.e. 'my_top_100' CSV becomes 'My Generated' Spotify playlist)
        new_playlist = sp.user_playlist_create(username, name=p_name, public=False) # Create playlist
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks) # Add the tracks to playlist                   
        print('{} playlist created successfully! Check Spotify!'.format(p_name))
        
        # Print how many of the selected tracks are from each source playlist
        selected = self.data.loc[self.data['id'].isin(tracks)].groupby('source') \
            .count().sort_values('id', ascending=False)['id']
        print('Source Playlist\tCount')
        print('\t'.join(selected.to_csv().split(',')))
        
    """
    Tests out various hidden layer sizes and initial random_states in order to find a MLP classifier
    with the highest training score. 
    
    PARAMS:
         features [Optional] - Features to use in training
    RETURNS:
         None, sets self.mlp_classifier to the best performing model
    
    """
    def benchmark_mlp(self, features=['energy', 'liveness', 'speechiness', 
                                      'acousticness', 'instrumentalness', 'danceability', 
                                      'valence', 'loudness_0_1', 'tempo_0_1']):
        # Make sure any included generated columns have been created before starting
        # i.e. if GMM clusters are to be used in classification but have not been generated yet
        self.check_features(features)

        # Create X and Y matrices
        X = self.data[features] # X is the selected features for each track
        Y = self.data[['source']].values.ravel() # Y is the source playlist each track came from

        best_score = (0,(0,),0) # Best trial: (score, layers, seed)
        layer_options = [(150,200,100),(30,200,50),(150,250,350,200,100),(15,),(10,),(20,),
                         (150,200,100,300,250,125)]
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
                                            random_state=best_score[2],
                                            shuffle=True, 
                                            tol=1e-4)
        self.mlp_classifier.fit(X,Y)      
    
    
    """
    Performs classification using a Gaussian Naive Bayes classifier. Seems to be less accurate than
    using the MLP classifier in the above functions.
    
    PARAMS:
         features [Optional] - The features to be used in X values in training
    RETURNS:
         None
    
    """ 
    def classify_naive_bayes(self, features=['energy', 'liveness', 'speechiness', 
                                            'acousticness', 'instrumentalness', 'danceability', 
                                            'valence', 'loudness_0_1', 'tempo_0_1']):
        
        # Make sure any included generated columns have been created before starting
        self.check_features(features)
        
        # Create X and Y matrices
        X = self.data[features] # X is the selected features for each track
        Y = self.data[['source']].values.ravel() # Y is the source playlist each track came from
        
        # Create Naive Bayes, train, predict from X and see how many are correctly matched        
        nb = GaussianNB()
        nb.fit(X,Y)
        predict_Y = nb.predict(X)
        print('{} classified correctly out of {}.'.format(np.sum(Y == predict_Y), len(Y)))
        # Note: This performs worse than the MLPClassifier
     
     
    # End of Classification Functions
    #=================================================================================
    # Start of Dance Party and Study Buddies Functions
    

    """
    Creates a playlist consisting of the N most danceable songs from each playlist passed to the function.
    First combines danceability and energy features then picks to top N from each playlist. Randomly picks
    from n songs from the top n + n/2 options, so different songs are picked each time even with the same 
    input playlists.

    PARAMS:
         playlists [Optional] - The array of playlist CSV filenames to use, defaults to 'all' to use all
         n         [Optional] - The number of tracks to use from each playlist, defaults to 3
    RETURNS:
         None (Playlist is created in Spotify with the name "Dance Party! [Today's Date]")
    
    """
    def dance_party(self, playlists='all', n=5):        
        if(playlists == 'all'):
            playlists = self.keys      

        # Add a column dance_value that is a combination of danceability and energy
        self.data['dance_value'] = self.data['danceability'] + self.data['energy']
        tracks = []
        for playlist in playlists:
            df = self.data.loc[self.data['source'] == playlist] # Select the songs from that playlist
            df = df.sort_values('dance_value', ascending=False) # Sort by the dance_value
            options = df.head(int(n + n/2))['id'].values.tolist() # Top int(n + n/2) tracks are options
            np.random.shuffle(options) # Shuffle so it picks slightly different songs each time
            tracks += options[:n] # Take n from the int(n+n/2) choices
        tracks = list(set(tracks)) # Remove duplicates
        np.random.shuffle(tracks) # Shuffle playlist order
        # TODO: Temporal analysis to try to make tracks flow into each other better?
        p_name = 'Dance Party! ' + datetime.now().strftime('%m/%d/%y') # Playlist name
        new_playlist = sp.user_playlist_create(username, name=p_name, public=False) # Create playlist
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks) # Add the tracks to playlist
        print("'{}' playlist created successfully!".format(p_name))


    """
    Creates a playlist consisting of the N songs from each playlist that would be best for a study session.
    Uses several audio features: instrumentalness, acousticness, energy, speechiness

    PARAMS:
         playlists [Optional] - The array of playlist CSV filenames to use, defaults to 'all' to use all
         n         [Optional] - The number of tracks to use from each playlist, defaults to 5
    RETURNS:
         None (Playlist is created in Spotify with the name "Study Buddies - [Today's Date]")
    
    """
    def study_buddies(self, playlists='all', n=5):
        if(playlists == 'all'):
            playlists = self.data.keys()

        # Add a column for study_value, combining 4 audio features that work pretty well for a quiet
        # study playlist that is conducive to studying while listening to music. (Based on testing and 
        # my own personal opinion of what makes good study music)
        self.data['study_value'] = (2 * self.data['instrumentalness']) + self.data['acousticness'] + \
            (2 * (1 - self.data['energy'])) + (1 - self.data['speechiness'])
        tracks = []
        for playlist in playlists:
            df = self.data.loc[self.data['source'] == playlist] # Select the songs from that playlist
            df = df.sort_values('study_value', ascending=False) # Sort by the study_value
            options = df.head(int(n + n/2))['id'].values.tolist() # Top int(n + n/2) tracks are options
            np.random.shuffle(options) # Shuffle so it picks slightly different songs each time
            tracks += options[:n] # Take n from the int(n+n/2) choices
        tracks = list(set(tracks)) # Remove duplicates
        np.random.shuffle(tracks) # Shuffle playlist order        
        # TODO: Temporal analysis to try to make tracks flow into each other better?    
        p_name = 'Study Buddies - ' + datetime.now().strftime('%m/%d/%y') # Playlist name
        new_playlist = sp.user_playlist_create(username, name=p_name, public=False) # Create playlist
        sp.user_playlist_add_tracks(username, new_playlist['id'], tracks) # Add the tracks to playlist
        print("'{}' playlist created successfully!".format(p_name))


    # End of Dance Party and Study Buddies Functions
    #=================================================================================
    # Start of Clustering Functions


    """
    Performs Gaussian mixture model clustering on the data. Varies the number of clusters 
    in order to find the best fitting clustering.

    PARAMS:
        show_plot [Optional] - Boolean, whether or not to call plot_clusters after, defaults to True
        features  [Optional] - The features to use in the clustering
    RETURNS:
        None, sets the 'GMM' column in self.data to the cluster indices from the optimal clustering
    
    """         
    def gmm_cluster(self, show_plot=True, n_clusters=5,
                    features=['energy', 'liveness', 'tempo_0_1', 'speechiness', 
                              'acousticness', 'instrumentalness', 'danceability', 
                              'loudness_0_1', 'valence']):
        
        self.check_features(features) # Make sure features are valid/generated already if applicable
        print('Starting GMM clustering with {} clusters and features = {}'.format(n_clusters, features))
        
        X = self.data[features]        
        gm = GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter=300, n_init=5)
        gm.fit(X)
        Y = gm.predict(X)
        self.data['GMM'] = Y

        if show_plot:
            self.plot_clusters(features=features, algorithm='GMM')

    
    """
    Benchmarks GMM clustering by varying the number of clusters used to find which clustering 
    fits the data best.
   
    PARAMS:
         show_plot [Optional] - Show the graphs after completion, defaults to True
         features  [Optional] - The features to use in clustering
    RETURNS:
         None, but shows plots in a new window (one at a time) is show_plots is set to True
    
    """
    def benchmark_gmm_cluster(self, show_plot=True, 
                              features=['energy', 'liveness', 'tempo_0_1', 'speechiness', 
                                        'acousticness', 'instrumentalness', 'danceability', 
                                        'loudness_0_1', 'valence']):
        self.check_features(features) # Make sure features are valid/generated already if applicable
        print('Benchmarking GMM clustering with features = {}'.format(features))
        
        X = self.data[features]        
        best_score = None  
        best_n = None
        
        for n in range(2,15): # Find the optimal number of components
            gm = GaussianMixture(n_components=n, covariance_type='full', max_iter=300, n_init=5)
            gm.fit(X)
            score = gm.bic(X)
            if best_score == None or score < best_score:
                best_score = score
                best_n = n
                best_gm = gm
        
        print('Best Score: {} with N = {}'.format(best_score, best_n))
        Y = best_gm.predict(X)
        self.data['GMM'] = Y

        if show_plot:
            self.plot_clusters(features=features, algorithm='GMM')

    """
    Performs spectral clustering on all the data, using n clusters.

    PARAMS:
         show_plot  [Optional] - Boolean, whether or not to call plot_clusters after, defaults to True
         n_clusters [Optional] - The number of clusters to use, defaults to 5
         features   [Optional] - The features to use in the clustering
    RETURNS:
         None, sets the 'spectral' column in self.data to the predicted clusters
    
    """
    def spectral_cluster(self, show_plot=True, n_clusters=5,
                         features=['energy', 'liveness', 'speechiness', 
                                   'acousticness', 'instrumentalness', 'danceability', 
                                   'valence', 'loudness_0_1', 'tempo_0_1']):
       
        self.check_features(features) # Make sure features are valid/generated already if applicable
        print('Starting Spectral clustering with {} clusters and features = {}'.format(n_clusters, features))
        
        X = self.data[features]
        spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                      affinity='nearest_neighbors', n_init=5)
        Y = spectral.fit_predict(X)
        self.data['spectral'] = Y

        if show_plot:
            self.plot_clusters(features=features, algorithm='spectral')
   
    """
    Plots the clusters (in different colors)
         - First a bar chart showing each playlists' distributions of clusters
         - Then, for each combination of features (2D), shows all playlists' data combined on a scatter plot
            with each cluster in a different color. 
    
    PARAMS:
        algorithm  [Optional] - {'GMM' or 'spectral'} The clustering algorithm you want to plot
        x          [Optional] - The feature for the x axis, default to None which cycles through all features
        y          [Optional] - The feature for the y axis, default to None which cycles through all features
        features   [Optional] - The list of features to plot, each used as the x and y axis with all others
    RETURNS:
        None, displays plots in a new window (one at a time)
    
    """
    def plot_clusters(self, algorithm='GMM', x=None, y=None,
                      features=['energy', 'liveness', 'tempo_0_1', 'speechiness', 
                                'acousticness', 'instrumentalness', 'danceability', 
                                'loudness_0_1', 'valence']):
        
        # Make sure clustering has been run already and features valid
        self.check_features(features + [algorithm])

        n_clusters = self.data[algorithm].max() + 1 # Find the number of clusters used

        # Bar chart - Cluster count for each file (each playlist)
        fig, ax = plt.subplots()
        counts = np.zeros(shape=(len(self.sources), n_clusters))
        i = 0
        file_names = self.data.keys()
        for f,data in self.data.groupby('source'):
            for cluster, group in data.groupby([algorithm]):
                counts[i,cluster] = len(group)
            i += 1
        plt.title('Cluster Frequencies in Each Playlist')
        fig.canvas.set_window_title('Cluster_Frequencies_in_Each_Playlist_{}_{}'.
                                    format(algorithm, n_clusters))
        counts_df = pd.DataFrame(counts)
        counts_df.plot.bar(legend=None, ax=ax, fig=fig)
        # Label the x axis values with the source names (just the part before the first _)   
        plt.xticks(range(len(self.sources)), map(lambda s: s.split('_')[0].title(), self.sources))  
        plt.xlabel('Playlist')
        plt.ylabel('Number of Tracks in Each Cluster')
        plt.show()

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
                    for name, group in self.data.groupby([algorithm]):
                        ax = group.plot.scatter(ax=ax, x=x_feature, y=y_feature,
                                                alpha=0.7, label=name, c=next(colors), legend=None)
                    plt.xlabel(x_feature.title())
                    plt.ylabel(y_feature.title())
                    plt.title(y_feature.title() + ' versus ' + x_feature.title())
                    fig.canvas.set_window_title('{}_vs_{}_{}_{}'.format(y_feature.title(), 
                                                            x_feature.title(), algorithm, n_clusters))
                    plt.show()


    # End of Clustering Functions
    #=================================================================================

    
    """
    Finds the artists who occur the most frequently in the given playlist. Takes into account the fact that
    some songs have multiple artists. Prints the top n occurring artists (or all if no n is given) and how 
    many tracks of theirs are in the playlist. 
    
    Note: This doesn't use any of the instance variables of the class (since the CSV's only have the primary
    artist for each song) but it is included in this class since it is still analysis of a playlist.
    
    PARAMS:
         playlist     - The Spotify playlist object (full) or Playlist ID or URI
         n [Optional] - The number of artists to display, defaults to None to print all artists present
    RETURNS:
         None, just prints results to stdout
    
    """
    def top_artists(self, playlist, n=None):
        
        # In case the ID/URI is passed (as opposed to the playlist object)
        if type(playlist) is str:
            playlist = sp.user_playlist(username, playlist.split(':')[0])
        
        print('Finding the top artists in {}...\n'.format(playlist['name']))
    
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
    
        # If n was not set (defaults to None) then print all artists
        if n is None:
            n = len(artists_map.keys())
            print('Top artists in playlist:')
        else:
            print('Top {} artists in playlist:'.format(n))    
        # Print them in order in a column format (I wish CCR picked a shorter name!)
        for (num, artist) in heapq.nlargest(n, heap):
            print("{1:<8} -  {0:<30}".format(artist[:30],  # Only first 30 characters of song name
                                       str(num) + (" song " if num==1 else " songs")))  # Plural if >1
    
        # Diversity (can be >100 because songs can have multiple artists)
        print("\n{} different artists appeared in this playlist.\n".format(len(artists_map.keys())))    
        
    
    """
    Helper function, make sure the feature is a valid option and has been generated already 
    if it needs to be (i.e. clusters)
    
    PARAMS:
         features - The features to check
    RETURNS:
         None, raises an error if there is a problem
    
    """
    def check_features(self, features):
        for f in features:
            if f not in self.data.keys():
                features.remove(f)
                if f == 'GMM':
                    self.gmm_cluster(show_plot=False, features=features)
                elif f == 'spectral':
                    self.spectral_cluster(show_plot=False, features=features)
                else:
                    raise KeyError('Error: Invalid feature ({})\nPossible ' + \
                                     'features are: GMM, spectral, {}'.format(f, self.data.keys()))
    
    """
    Helper function, makes sure the playlist is a valid option

    PARAMS:
         playlist - The playlist csv filenmae to check
    RETURNS:
         None, raises an error if there is a problem
    
    """    
    def check_playlist(self, playlist):
        if playlist not in self.sources:
            raise KeyError("Error: Playlist '{}' not found in list of playlists ({})\n"+\
                             "Try using '-w -p PLAYLIST_ID' to write a CSV first and make sure "+\
                             "not to include '.csv' in the name.".
                             format(playlist, self.sources))        
   
 
#===============================================================================   
# ------------------- End of Analyzer Class ------------------------------------
#===============================================================================   
# ------------------- Start of Writer Class ------------------------------------
#===============================================================================   

"""
The Writer class provides two functions: write_csv and write_all_csvs

The first takes a Spotify playlist object and writes a csv file containing the audio features
and metadata for each track in the playlist.

The second searches the user's (defined in SPOTIFY_USERNAME environment variable) playlists for 
any that contain Top 100 (since I have acquired 12 different Top 100 playlists from different people)
and writes a csv file for each.
"""
class Writer:
        
    """
    Writes a CSV file for the passed playlist that contains the metadata and audio features for each 
    track in the playlist.
    
    The playlist 'My Top 100' gets a CSV file written for it named 'my_top_100.csv' in the csv directory.
    
    PARAMS:
         playlist - Spotify playlist object (full) to be analyzed or Playlist ID/URI
    RETURNS:
         None
    
    """
    def write_csv(self, playlist):
        if type(playlist) is str: # Was given ID/URI not the Playlist object
            playlist = sp.user_playlist(username, playlist.split(':')[0]) # Get playlist object
    
        outfile = 'csv/' + playlist['name'].replace(' ', '_').lower() + '.csv'
    
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
        offset = 0 # Need to use batches since the API endpoint has a limit of 50 tracks
        all_tracks_features = []
        while offset < len(metadata.keys()): # This works for playlist of any size, not just 100
            all_tracks_features += sp.audio_features(metadata.keys()[offset:offset+50]) # Limit 50 tracks for this endpoint
            offset += 50

        # Open the CSV (rewrites old data, creates if doesn't exist)
        writer = csv.writer(open(outfile, 'w+')) 
    
        header_row = ['Track Name', 'Primary Artist', 'Album'] # The header row of the CSV, first the Metadata
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
    Writes a CSV for every playlist that has "Top 100" in the name for the user (defined in SPOTIFY_USERNAME 
    environment variable). I gathered a bunch of Top 100 playlists from friends and this way I can just 
    write all the CSV's in one function call to make sure they're all present and up to date. 
    
    PARAMS:
         None
    RETURNS:
         None
    
    """
    def write_all_csvs(self):
        all_playlists = sp.current_user_playlists(limit=50) # Returns simplified playlist objects
        selected_playlists = []
        for playlist in all_playlists['items']:
            if 'Top 100' in playlist['name']:
                selected_playlists.append(playlist)
        for p in selected_playlists:
            p_uri = p['uri'].split(':')[-1] # Get the ID of the playlist
            playlist = sp.user_playlist(username, p_uri) # Get the full playlist object
            self.write_csv(playlist)



#===============================================================================
# -------------------------    End of Writer Class   ---------------------------
#===============================================================================   
# -------------------------   Start of Unit Testing   --------------------------
#===============================================================================   


"""
TestAnalyzer class uses unittest to test the various features of Writer and Analyzer

Note: All tests are skipped right now since they have been tested already. 

"""
skip_tests = False
skip_tests = True # Comment this line out to run all the tests (disable skips)
class TestAnalyzer(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        # runs once before ALL tests
        print('\n.... starting unit testing of playlist_analyzer.py ...')
            
    @unittest.skipIf(skip_tests, '')
    def test_top_artists(self):
        print('\n............ testing top artists ..................')
        # Call with the Playlist object and only display top 15 artists
        playlist_id = '38hwZEL0H1z6wUbq0UoHBS'
        playlist = sp.user_playlist(username, playlist_id)
        analyzer.top_artists(playlist, n=10)
        
        # Call with the string instead, and with no n given (so defaults to all artists)
        analyzer.top_artists(playlist_id)
    
    @unittest.skipIf(skip_tests, '')
    def test_predict_mlp(self):
        print('\n........... testing predict MLP ...................')
        analyzer.predict_playlist(playlist='my_top_100')
    
    @unittest.skipIf(skip_tests, '')
    def test_benchmark_mlp(self):
        print('\n........... testing benchmark MLP .................')
        analyzer.benchmark_mlp()
     
    @unittest.skipIf(skip_tests, '')
    def test_dance_party(self):
        print('\n............ testing dance party ..................')
        analyzer.dance_party(playlists=['p_top_100', 'a_top_100', 'ri_top_100', 'm_top_100'])    
    
    @unittest.skipIf(skip_tests, '')
    def test_study_buddies(self):
        print('\n.......... testing study buddies ..................')
        source_playlists = ['rik_top_100', 'j_top_100', 'am_top_100', 'm_top_100', 'c_top_100']
        analyzer.study_buddies(playlists=source_playlists, n=6)
    
    @unittest.skipIf(skip_tests, '')
    def test_invalid_playlist(sef):
        print('\n.......... testing invalid playlist ...............')
        try:
            # Should use csv filename for this function (c_top_100) not ID
            analyzer.predict_playlist(playlist='5kipTqpcNptT9sCcrV0RdW')
        except Exception as e:
            print(e)
        print('\n(Expected an error)')
        
    @unittest.skipIf(skip_tests, '')
    def test_invalid_feature(self):
        print('\n.......... testing invalid feature ................')
        try:
            # Synergy is not a valid audio feature
            analyzer.gmm_cluster(features=['energy','synergy'])
        except Exception as e:
            print(e)
        print('\n(Expected an error)')
   
    @unittest.skipIf(skip_tests, '')
    def test_gmm_clustering(self):
        print('\n.......... testing gmm clustering ..................')
        
        # Just use 2 features
        print('Testing using only valence and energy...')
        analyzer.gmm_cluster(show_plot=True, features=['valence', 'energy'])
        
        # Use all features
        print('Testing using all features...')
        analyzer.gmm_cluster(show_plot=True)

    @unittest.skipIf(skip_tests, '')
    def test_plot_clusters(self):
        print('\n.......... testing cluster plotting .................')
        # First make sure some clusters have been generated
        analyzer.gmm_cluster(n_clusters=3, show_plot=False)
        analyzer.plot_clusters(algorithm='GMM', x='energy', y='valence') # Plot specific x,y combo only
        analyzer.plot_clusters(algorithm='GMM', y='valence') # Cycle through each x
        analyzer.plot_clusters(algorithm='GMM', x='energy')  # Cycle through each y
           
    @unittest.skipIf(skip_tests, '')
    def test_benchmark_gmm(self):
        print('\n...... testing benchmark GMM clustering ...........')
        
        # Use all features - then show all plots
        print('Benchmarking GMM clustering using all features...')
        analyzer.benchmark_gmm_cluster(show_plot=False)
        analyzer.plot_clusters(algorithm='GMM', y='valence')
                
        # Only use energy and valence - then show plots
        print('Benchmarking GMM clustering using only 2 features...')  
        analyzer.benchmark_gmm_cluster(show_plot=True, features=['energy','valence'])
    
    #@unittest.skipIf(skip_tests, '')
    def test_spectral_clustering(self):
        print('\n........... testing spectral clustering ...........')
        #analyzer.spectral_cluster()        
        analyzer.spectral_cluster(features=['energy','valence'])
        
    def tearDown(self):
        print('\n')
      
    @classmethod
    def tearDownClass(self):
        print('\n.... finished unit testing of playlist_analyzer.py ...')
        print('\n\nUse -h or --help flag from command line to see how to use '+\
              '\neach feature if you did not mean to perform unittesting.')
        
#===============================================================================   
# -------------------------     End of Unit Testing    -------------------------  
#===============================================================================   
# -------------------------     Start of Main Method    ------------------------ 
#===============================================================================

"""
Main Method - Connects to Spotify API, parses command line arguments and calls 
the appropriate function. If no command line arguments are given, unittest.main() is called.

-h or --help can be provided to print a help message on how to use the different features.

"""
if __name__ == '__main__':
    
    # Access enviornment variables for authenticating with Spotify
    try:
        username = os.environ['SPOTIFY_USERNAME']
        client_id = os.environ['SPOTIFY_CLIENT_ID']
        client_secret = os.environ['SPOTIFY_CLIENT_SECRET']
        redirect_uri = os.environ['SPOTIFY_REDIRECT_URI']
    except KeyError:
        print('\nThe following environment variables must be set in order to run this script:\n' +\
              '\t- SPOTIFY_USERNAME\n\t- SPOTIFY_CLIENT_ID\n\t- SPOTIFY_CLIENT_SECRET\n\t- '+\
              'SPOTIFY_REDIRECT_URI\n'+\
              '\nPlease see README.md or matt-levin.com/PlaylistAnalyzer for more details.\n')    
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
    analyzer = Analyzer()
    
    if(len(sys.argv) > 1): # Command line were arguments given
        feature = sys.argv[1] # Which feature is being used
        args = {} # The remaining arguments stored as a dict
        for i in range(2, len(sys.argv), 2): # Parse any optional arguments into args dict
            try:
                args[sys.argv[i]] = sys.argv[i+1]
            except IndexError:
                print('Invalid arguments, use -h or --help flag to see proper usage')
                sys.exit(0)
        
        # Generate Predicted Playlist with MLP
        if feature == '-g':
            playlist = 'my_top_100'
            if '-p' in args:
                playlist = args['-p'].split('.')[0] # Strip .csv if given
            n = 100
            if '-n' in args:
                n = int(args['-n'])
            analyzer.predict_playlist(playlist, n)
        
        # Top Artists
        elif feature == '-a':
            n = 15 # Default to showing top 15 artists
            if '-n' in args:
                n = int(args['-n'])
            playlist_id = '38hwZEL0H1z6wUbq0UoHBS' # Default playlist to analyze
            if '-p' in args:
                playlist_id = args['-p'].split(':')[-1] # Isolate ID if more was given
            playlist = sp.user_playlist(username, playlist_id)
            analyzer.top_artists(playlist, n)
        
        # Write CSV of audio features for given playlist (or all user's Top 100 playlists if none provided)
        elif feature == '-w':
            writer = Writer()
            if '-p' in args: # Of a single playlist if -p argument provided
                playlist_id = args['-p'].split(':')[-1] # Isolate ID if more was given
                playlist = sp.user_playlist(username, playlist_id)
                writer.write_csv(playlist)
            else: # Default to all user playlists (that have 'Top 100' in the name)
                writer.write_all_csvs()
        
        # Dance Party
        elif feature == '-d':
            playlists = 'all'
            if '-p' in args:
                playlists = args['-p'].split(',') # Split on commas
                playlists = map(lambda s: s.split('.')[0], playlists) # Remove '.csv' if given
            n = 5
            if '-n' in args:
                n = int(args['-n'])
            analyzer.dance_party(playlists, n)
        
        # Study Buddies
        elif feature == '-s':         
            playlists = 'all'
            if '-p' in args:
                playlists = args['-p'].split(',') # Split on commas
                playlists = map(lambda s: s.split('.')[0], playlists) # Remove '.csv' if given
            n = 5
            if '-n' in args:
                n = int(args['-n'])
            analyzer.study_buddies(playlists, n)
            
        # Clustering
        elif feature == '-c':
            algorithm = 'GMM'
            if '-a' in args:
                algorithm = args['-a']
            n = None
            if '-n' in args:
                n = int(args['-n'])
            # For now it just uses all features, eventually that could be an argument as well
            if algorithm.lower() == 'gmm':
                if n: # If n_clusters was given, just use that
                    analyzer.gmm_cluster(show_plot=True, n_clusters=n)
                else: # If not, run benchmarking to find the best fitting n_clusters
                    analyzer.benchmark_gmm_cluster(show_plot=True)
            elif algorithm.lower() == 'spectral':
                if not n:
                    n = 5 # Default to 5 if not given
                analyzer.spectral_cluster(show_plot=True, n_clusters=n)
            else:
                raise ValueError("Algorithm must be either 'GMM' or 'spectral'")
            
        # Show help message
        elif feature == '-h' or feature == '--help':
            print("""
This project contains several functions, use one of the following choices and its [optional] parameters:
Please see README.md or matt-levin.com/PlaylistAnalyzer for more details.\n
-h or --help\n\tShows this help message\n
-g [-p Playlist_CSV_Filename] [-n Number_Of_Songs_To_Include]
\tGenerates a playlist similar to the given playlist using a neural network\n
-a [-p Playlist_ID] [-n Number_Of_Artists_To_Display]
\tShows top occuring n Artists in a Playlist (defaults to showing all artists in order)\n
-w [-p Playlist_ID]
\tWrite CSV file of audio features for playlist (defaults to all Top 100 playlists)\n
-d [-n Number_Songs_From_Each_Playlist] [-p Playlist_CSV_Names_Seperated_By_Commas]
\tGenerates a Dance Party playlist from given Playlist CSV file names, taking n songs from each\n
-s [-n Number_Songs_From_Each_Playlist] [-p Playlist_CSV_Names_Seperated_By_Commas]
\tGenerates a study playlist from given Playlist CSV file names, taking n songs from each\n
-c [-a Algorithm] [-n Number_Of_Clusters]
\tPerforms clustering with given algorithm {'GMM' or 'spectral'} and number of clusters\n""")

        # Unrecognized feature selection
        else:
            print("Invalid selection, please use '-h' or '--help' to show correct usage.")
    

    # Run unit testing if no arguments were given
    else:
        print('Use -h or --help flag to see how to use each feature from Command Line.\n'+\
             'Beginning unittesting since no arguments were given...')
        unittest.main()

    #print('\nProgram Complete')
