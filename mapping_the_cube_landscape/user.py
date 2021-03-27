from collections import defaultdict
import pandas as pd
import numpy as np
import itertools
import scipy.stats as stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from gap_statistic import OptimalK
import umap.umap_ as umap
import seaborn as sns
import random as random
import re


def create_cooccurence_matrix(decklists, drop_basics=True, card_subset=None):
    '''Creates a co-occurence matrix for all cards present in the maindeck
        Params:
        -------
        drop_basics: boolean, default True. Whether or not basics should be included in
            co-occurence matrix
        card_subset: list of card strings, will subset the cooccurence matrix to these cards'''
    
    deck_list = [deck.main for deck in decklists]

    # create card index, defaults to alphabetical
    card_list = set(list(itertools.chain(*deck_list)))
    card_to_id = dict(zip(card_list, range(len(card_list))))

    # convert decks to indices
    decks_as_ids = [np.sort([card_to_id[card] for card in deck]).astype('uint32') for deck in deck_list]

    # extract data for rows and columns
    row_ind, col_ind = zip(*itertools.chain(*[[(i, card) for card in deck] for i, deck in enumerate(decks_as_ids)]))

    # create matrix and identify largest index
    data = np.ones(len(row_ind), dtype='uint32')
    max_card_id = max(itertools.chain(*decks_as_ids)) + 1

    # efficient arithmetic operations with CSR * CSR
    deck_card_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(decks_as_ids), max_card_id))

    # multiplying deck_card_matrix with its transpose generates the co-occurences matrix
    card_cooc_matrix = deck_card_matrix.T * deck_card_matrix

    # coocurence of diagonals is 0
    card_cooc_matrix.setdiag(0)

    # create pandas dataframe summarizing matrix
    card_matrix = pd.DataFrame(card_cooc_matrix.todense(), index = card_to_id.keys(), columns = card_to_id.keys())

    if card_subset:
        missing_entries = set(card_subset) - set(card_matrix.index)
        if missing_entries:
            raise ValueError('{} cards not found in card_matrix'.format(missing_entries))
        else:
            card_matrix = card_matrix.loc[card_subset, card_subset]
        
    # drop basic lands
    if drop_basics:
        basics = ['Mountain', 'Forest', 'Island', 'Plains', 'Swamp']
        card_matrix = card_matrix.drop(index = basics, columns = basics)

    return card_matrix

class User:

    '''A class to store data about cubetutor users'''
    def __init__(self, cubeid):
    
        # attributes related to the cube
        self.cubeid = cubeid
        self.shortid = None
        self.user = None
        self.cube_name = None
        self.cube_size = None
        self.cube_category = None
        self.cube_list = None
        self.last_updated = None
        self.percentage_fixing = None
        self.cube_image = None
        self.num_followers = None
        self.islisted = None
        
        # data attributes of the cube
        self.decks = []
        self.card_data = defaultdict(lambda: {'main': 0, 'side': 0})
        self.deck_df = None
        
        # attributes related to PCA
        self.pca_model = None
        self.pca_result = None
        
        # attributes related to tSNE
        self.tsne_model = None
        self.tsne_result = None
        
        # attributes related to UMAP
        self.umap_model = None
        self.umap_result = None
        
        # attributes related to kmeans
        self.kmeans_model = None
        self.kmeans_result = None
        self.kmeans_predictions = None
        
        # attributes related to optimal K for kmeans
        self.optimal_k_obj = None
        self.optimal_cluster_num = None
        
        # cluster specific data
        self.cluster_card_df = None
        self.card_significance_df = None
        self.cluster_maindeck_df = None

    def extract_card_data(self, built_filter = True, draft_filter = True):
        '''From the user's decks, extract statistics about the cards'''

        for deck in self.decks:
            if draft_filter:
                if built_filter:
                    if deck.built and deck.draft:
                        for card in deck.main:
                            self.card_data[card]['main'] += 1
                        for card in deck.side:
                            self.card_data[card]['side'] += 1
                else:
                    if deck.draft:
                        for card in deck.main:
                            self.card_data[card]['main'] += 1
                        for card in deck.side:
                            self.card_data[card]['side'] += 1
            else:
                for card in deck.main:
                    self.card_data[card]['main'] += 1
                for card in deck.side:
                    self.card_data[card]['side'] += 1
        
        return self
    
    def extract_maindecks(self, built_filter = True, draft_filter = True, min_play_filter = True):
        '''Builds a dataframe of decks, n_decks x n_cards
        
        Excludes decks that are not built or are likely sealed decks
        '''
        
        # extract the decks that are built and not sealed decks
        if draft_filter: 
            if built_filter:
                built_draft_decks = [(i, deck) for i, deck in enumerate(self.decks) if deck.built and deck.draft]
            else:
                built_draft_decks = [(i, deck) for i, deck in enumerate(self.decks) if deck.draft]
        else:
            built_draft_decks = [(i, deck) for i, deck in enumerate(self.decks)]

        indices, built_decks = zip(*built_draft_decks)
        
        # extract cards that are in maindecks
        columns = [card for card, value in self.card_data.items() if value['main'] > 0]
        
        # build the dataframe to fill in
        deck_df = np.zeros(shape = (len(built_decks), len(columns)))
        card_index_lookup = dict(zip(columns, range(len(columns))))
        
        # fill in the deck df, adding a 1. Adding 1 handles non-singleton cubes.
        for i, deck in enumerate(built_decks):
            for card in deck.main:
                card_index = card_index_lookup[card]
                deck_df[i, card_index] += 1
            
        # convert to a dataframe
        deck_df = pd.DataFrame(deck_df, index = indices, columns = columns)
        
        # an ugly hack to remove columns for basic lands
        basic_re_matches = [(col, re.search(col, '^Mountain$|^Island$|^Swamp$|^Plains$|^Forest$', re.IGNORECASE)) for col in deck_df.columns]
        snow_basic_re_matches = [(col, re.search(col, '^Snow-Covered Mountain$|^Snow-Covered Island$|^Snow-Covered Swamp$|^Snow-Covered Plains$|^Snow-Covered Forest$', re.IGNORECASE)) for col in deck_df.columns]
        basic_cols = [card for card, match in basic_re_matches if match is not None]
        snow_basic_cols = [card for card, match in snow_basic_re_matches if match is not None]
        deck_df = deck_df.drop(columns = basic_cols + snow_basic_cols)
        
        # some decks weirdly have way too many basics (eg worldknit. Filter these out.)
        if min_play_filter:
            bad_decks = deck_df[deck_df.sum(axis = 1) < 18].index
            deck_df = deck_df.drop(index = bad_decks)
        
        if 'invalid card' in deck_df.columns:
            deck_df = deck_df.drop(columns = ['invalid card'])

        self.deck_df = deck_df

        return deck_df
    
    def run_pca(self, n_components=100, verbose=False, **pca_kwargs):
        '''Runs PCA on the deck dataframe'''
        
        pca_model = PCA(n_components = n_components, **pca_kwargs)
        pca_result = pca_model.fit_transform(self.deck_df)
        pca_df = pd.DataFrame(pca_result, columns = ['PCA_{}'.format(i + 1) for i in range(n_components)],
                              index = self.deck_df.index)
        
        self.pca_model = pca_model
        self.pca_result = pca_df
        
        if verbose:
            print('Explained variance for {} principal components: {}'.format(n_components, np.sum(pca_model.explained_variance_ratio_)))
        
        return pca_result
    
    def run_tsne(self, pca_components = None, **tsne_kwargs):
        '''Run tsne on the PCA result, with a number of PCA components as shown.
        By default, uses all the PCA components'''
         
        tsne_model = TSNE(n_components=2, **tsne_kwargs)
        
        # subset pca data if necessary
        if pca_components is None:
            pca_components = self.pca_result.shape[1]
        pca_data = self.pca_result.values[:, :pca_components]
        
        # run on pca data
        tsne_result = tsne_model.fit_transform(pca_data)
        tsne_result = pd.DataFrame(tsne_result, columns = ['tSNE_1', 'tSNE_2'])
        
        self.tsne_model = tsne_model
        self.tsne_result = tsne_result
        
        return tsne_result
    
    def run_kmeans(self, n_clusters, pca_components = None, **kmeans_kwargs):
        '''Run kmeans on the PCA result'''
        
        kmeans_model = KMeans(n_clusters = n_clusters, **kmeans_kwargs)
        
        # subset pca data if necessary
        if pca_components is None:
            pca_components = self.pca_result.shape[1]
        pca_data = self.pca_result.values[:, :pca_components]
        
        # fit the data, then run the clustering
        kmeans_result = kmeans_model.fit(pca_data)
        kmeans_predictions = kmeans_model.predict(pca_data)
        
        self.kmeans_model = kmeans_model
        self.kmeans_result = kmeans_result
        self.kmeans_predictions = kmeans_predictions
        
        return kmeans_predictions
    
    def reorder_clusters(self, projection='tsne'):
        '''A helper function that just rearranges clusters so that clusters are labelled
        0, 1, 2...n as a function of centroid distance from the origin based on the 2D projection
        given. Updates self.kmeans_predictions'''
        
        # extract projection results
        if projection.lower() == 'tsne':
            projection_results = self.tsne_result.copy()
        elif projection.lower() == 'umap':
            projection_results = self.umap_result.copy()
        else:
            raise ValueError('Unrecognized projection')
        
        # calculate the centroids
        projection_results['cluster'] = self.kmeans_predictions
        projection_centroids = projection_results.groupby('cluster').mean()
        
        # calculate pairwise distances between clusters
        pairwise_dist = pd.DataFrame(squareform(pdist(projection_centroids)))
        
        # calculate the bottom-left-most centroid
        bottom_left_cluster = projection_centroids.mean(axis = 1).idxmin()

        # iteratively find the closest cluster to the current one
        current_cluster = bottom_left_cluster
        cluster_mapping = {}

        for i in range(len(pairwise_dist.columns)):

            # identify the closest cluster. First extract distance to all clusters
            cluster_dist = pairwise_dist[current_cluster].sort_values()
            nonchosen_clusters = [clust for clust in cluster_dist.index if clust not in cluster_mapping]

            # the closest cluster is the first one in this list
            current_cluster = nonchosen_clusters[0]
            cluster_mapping[current_cluster] = i
        
        # perform the mapping, then overwrite
        new_kmeans_predictions = [cluster_mapping[clust] for clust in self.kmeans_predictions]
        self.kmeans_predictions = new_kmeans_predictions
        
        return new_kmeans_predictions
        
    def determine_optimal_k(self, pca_components = None, cluster_array = range(20, 60), n_refs = 10, **optimalk_kwargs):
        '''Uses the gap statistic (https://github.com/milesgranger/gap_statistic)
        to determine the optimmal number of clusters. Often does not work.
        
        cluster_array refers to the number of clusters that are tested.
        '''
        
        optimalK = OptimalK(**optimalk_kwargs)
        
        # subset pca data if necessary
        if pca_components is None:
            pca_components = self.pca_result.shape[1]
        pca_data = self.pca_result.values[:, :pca_components]

        n_clusters = optimalK(pca_data, n_refs = n_refs, cluster_array = cluster_array)
        self.optimal_k_obj = optimalK
        self.optimal_cluster_num = n_clusters
        
        return n_clusters
    
    def run_umap(self, pca_components = None, **umap_kwargs):
        '''Runs UMAP on the PCA components (defaults to all principal components)'''
        
        umap_model = umap.UMAP(**umap_kwargs)
        
        # subset pca data if necessary
        if pca_components is None:
            pca_components = self.pca_result.shape[1]
        pca_data = self.pca_result.values[:, :pca_components]
        
        umap_result = umap_model.fit_transform(pca_data)
        umap_result = pd.DataFrame(umap_result, columns = ['UMAP_1', 'UMAP_2'])
        
        self.umap_model = umap_model
        self.umap_result = umap_result
        
        return umap_result
    
    def plot_tsne_umap(self, ax, data_type, clusters = None, 
                       palette = None, cluster_highlight = None, noncluster_alpha = 0.02, **scatter_kwargs):
        '''Overlays the kmeans cluster data on umap or tsne and plots'''
        
        if scatter_kwargs is None:
            scatter_kwargs = {'linewidth': 0.4, 's': 20}

        if data_type == 'tSNE':
            x_col, y_col = 'tSNE_1', 'tSNE_2'
            data = self.tsne_result.copy()
        elif data_type == 'UMAP':
            x_col, y_col = 'UMAP_1', 'UMAP_2'
            data = self.umap_result.copy()
        else:
            raise ValueError('Unrecognized data type')
        
        # if clusters aren't passed, get them from kmeans
        if clusters is None:
            clusters = self.kmeans_predictions.copy()
            data['cluster'] = clusters
        else:
            data['cluster'] = clusters

        # make a color palette, handle -1 for hdbscan
        if palette is None:
            hls = sns.color_palette("hls", np.max(clusters) + 1)
            random.shuffle(hls)
            palette = dict(zip(range(len(hls)), hls))
            if -1 in data['cluster'].to_list():
                palette[-1] = 'grey'
        
        if cluster_highlight is None:
            sns.scatterplot(x = x_col, y = y_col, data = data, hue = 'cluster', 
                        palette = palette, ax = ax, **scatter_kwargs)
            
        else:
            cluster_data = data[data['cluster'] == cluster_highlight]
            noncluster_data = data[data['cluster'] != cluster_highlight]
            
            ax = sns.scatterplot(x = x_col, y = y_col, data = cluster_data, hue = 'cluster', 
                            palette = palette, ax = ax, **scatter_kwargs)
            ax = sns.scatterplot(x = x_col, y = y_col, data = noncluster_data, hue = 'cluster', 
                            palette = palette, alpha = noncluster_alpha, ax = ax, **scatter_kwargs)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        sns.despine(bottom = True, left = True, ax = ax)

        ax.get_legend().remove()
        
        return ax
    
    def determine_cluster_card_counts(self):
        '''Uses the clusters defined by kmeans to determine the card counts
        of each cluster. Used to identify cluster specific cards.'''
        
        # fisher's exact test does not work with non-binary (count) data
        binary_deck_df = self.deck_df.copy() > 0
        
        # index needs to match decks
        cluster_predictions = pd.Series(self.kmeans_predictions.copy())
        cluster_predictions.index = binary_deck_df.index
        num_clusters = cluster_predictions.max() + 1

        # first, for each cluster we sum the cards in those decks
        cluster_card_df = pd.DataFrame(index = range(num_clusters), columns = ['size'] + list(binary_deck_df.columns))
        for clust in range(num_clusters):
            deck_index = cluster_predictions.index[np.where(cluster_predictions == clust)]
            cluster_decks = binary_deck_df.loc[deck_index]

            # add together the counts in all the decks
            num_decks = cluster_decks.shape[0]
            cluster_card_counts = cluster_decks.sum(axis = 0)
            
            # store in matrix
            cluster_card_df.loc[clust, :] = [num_decks] + list(cluster_card_counts)
            
        self.cluster_card_df = cluster_card_df
        
        return cluster_card_df
    
    def determine_differential_cards(self, perc_filter = 0.1, verbose = False):
        '''Takes the cards from cluster_card_df and performs a pairwise fishers exact test
        comparing the cards in that cluster to all other clusters.'''
        
        card_significance_df = []
        for cluster, row in self.cluster_card_df.iterrows():
            if verbose: print(cluster, end = '..')
            
            # determine cluster size and cards in the cluster
            cluster_size = row[0]
            cards = row[1:]
            
            # only examine cards above filter
            represented_card_idx = cards > cluster_size * perc_filter
            represented_counts = cards[represented_card_idx]
            
            # for each other cluster, do a pairwise fishers exact test for each card.
            other_clusters = np.delete(np.arange(len(self.cluster_card_df)), cluster)
            for other_clust in other_clusters:
                other_clust_row = self.cluster_card_df.loc[other_clust]
                other_clust_size = other_clust_row[0]
                other_clust_cards = other_clust_row[1:]

                # get the counts for the represented cards in the other cluster
                other_represented_counts = other_clust_cards[represented_card_idx]

                # for each card, perform the fisher's exact test
                for card in represented_counts.index:
                    card_count_cluster = represented_counts.loc[card]
                    card_count_outcluster = other_represented_counts.loc[card]

                    cont_table = [[card_count_cluster, cluster_size - card_count_cluster],
                                  [card_count_outcluster, other_clust_size - card_count_outcluster]]
                    
                    try:
                        odds, p = stats.fisher_exact(cont_table, alternative = 'greater')
                    except:
                        print(card)
                        print(cont_table)
                        print(other_clust_size, card_count_outcluster)
                        print(other_represented_counts)
                        return other_clust_row
                    card_significance_df.append([cluster, other_clust, card, odds, p])
        
        card_significance_df = pd.DataFrame(card_significance_df, columns = ['cluster', 'other_cluster', 'card', 'odds', 'p'])
        card_significance_df['-logp'] = -np.log(card_significance_df['p'])/np.log(10)
        self.card_significance_df = card_significance_df
        return card_significance_df
    
    def plot_differential_cards(self, ax, cluster_idx, n_cards = 15, **stripplot_kwargs):
        '''Plots the differential cards for the cluster'''
        
        # get cluster data
        cluster_data = self.card_significance_df.query('cluster == {}'.format(cluster_idx)).copy()
        
        # order cards by p value
        ordered_cards = cluster_data.groupby('card')['-logp'].apply(lambda x: x.median()).sort_values(ascending = False).index[:n_cards]
        sns.stripplot(data = cluster_data, x = 'card', y = '-logp', ax = ax, order = ordered_cards, **stripplot_kwargs)
        
        # formatting
        sns.despine(ax = ax)
        #ax.set_title(' | '.join(ordered_cards[:8]))
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha = 'right')
        ax.set_xlabel('')
        
        return ax
    
    def determine_cluster_maindeck_rates(self, perc_filter = 0.2):
        '''Determines the maindeck rates of cards within clusters. perc_filter
        requires that the total number of times a card appears in a maindeck or sideboard
        exceeds perc_filter * cluster size'''
        
        cluster_maindeck_df = []
        
        # loop through the clusters
        for cluster in set(self.kmeans_predictions):
            
            # extract the decks that belong to that cluster
            cluster_deck_idx = np.where(np.array(self.kmeans_predictions) == cluster)
            cluster_decks = self.deck_df.index[cluster_deck_idx]
            cluster_decks = np.array(self.decks)[cluster_decks]
            
            # make a storage dictionary for main/side
            cluster_maindeck_data = defaultdict(lambda: {'main': 0, 'side': 0})
            
            # loop through decks in the cluster
            for clust_deck in cluster_decks:
                for card in clust_deck.main:
                    cluster_maindeck_data[card]['main'] += 1
                for card in clust_deck.side:
                    cluster_maindeck_data[card]['side'] += 1
            
            # turn this into a dataframe
            cluster_main_side_df = pd.DataFrame(cluster_maindeck_data).T
            
            # remove basics
            basic_re_matches = [(index, re.search(index, '^Mountain$|^Island$|^Swamp$|^Plains$|^Forest$', re.IGNORECASE)) for index in cluster_main_side_df.index]
            snow_basic_re_matches = [(index, re.search(index, '^Snow-Covered Mountain$|^Snow-Covered Island$|^Snow-Covered Swamp$|^Snow-Covered Plains$|^Snow-Covered Forest$', re.IGNORECASE)) for index in cluster_main_side_df.index]
            basic_cols = [card for card, match in basic_re_matches if match is not None]
            snow_basic_cols = [card for card, match in snow_basic_re_matches if match is not None]
            cluster_main_side_df = cluster_main_side_df.drop(index = basic_cols + snow_basic_cols)
            
            # calculate total and maindeck rates
            cluster_main_side_df['total'] = cluster_main_side_df['main'] + cluster_main_side_df['side']
            cluster_main_side_df['main %'] = cluster_main_side_df['main']/cluster_main_side_df['total']
            
            # remove cards that are not present at least perc_filter * cluster_size
            cluster_size = self.cluster_card_df.loc[cluster, 'size']
            cluster_main_side_df = cluster_main_side_df[cluster_main_side_df['total'] > cluster_size * perc_filter]
            
            # reset index and add cluster label
            cluster_main_side_df = cluster_main_side_df.reset_index()
            cluster_main_side_df = cluster_main_side_df.rename(columns = {'index': 'card'})
            cluster_main_side_df.insert(0, 'cluster', cluster)
            
            # append to storage dataframe
            cluster_maindeck_df.append(cluster_main_side_df)
            
        cluster_maindeck_df = pd.concat(cluster_maindeck_df)
        self.cluster_maindeck_df = cluster_maindeck_df
        return cluster_maindeck_df
        
        
    def plot_cluster_maindeck_rates(self, ax, cluster_idx, kind = None, n_cards = 15, **bar_kwargs):
        '''Plots the maindeck rates of cards within a cluster. kind refers to which cards - top is
        the most maindecked, bottom is the least, and both is a mixture'''
        
        # get cluster maindeck data
        cluster_maindeck_data = self.cluster_maindeck_df.query('cluster == {}'.format(cluster_idx)).copy()                                                           
        
        # order cards by maindeck %
        ordered_cards = cluster_maindeck_data.sort_values(by = 'main %', ascending = False)
        
        # subset down to specific cards based on user input
        if kind == 'bottom':
            ordered_cards = ordered_cards.tail(n_cards)
        elif kind == 'top':
            ordered_cards = ordered_cards.head(n_cards)
        elif kind == 'both':
            ordered_cards = pd.concat([ordered_cards.tail(n_cards // 2), ordered_cards.tail(n_cards // 2)])

        sns.barplot(data = ordered_cards, x = 'card', y = 'main %', ax = ax, **bar_kwargs)
        
        # formatting
        sns.despine(ax = ax)
        ax.set_xticklabels(ordered_cards['card'], rotation = 30, ha = 'right')
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel('')
        ax.set_ylim([0, 1.05])
        ax.grid(axis = 'y')
        
        return ax

    def sort_decklists(self):
        '''Overwrites the current deck attribute to one where the decklists are
           sorted in ascending date.'''
        
        dates = [deck.date for deck in self.decks]
        sorted_decks = [deck for _, deck in sorted(zip(dates, self.decks))]
        self.decks = sorted_decks
        
        return sorted_decks
    
    def calculate_total_coocurrence_matrix(self, drop_basics=True, n_filter=None):
        '''Calculates the cooccurence matrix using all cards in maindecks
        
        n_filter: int. Cards with fewer than n_filter will be excluded from the cooccurence matrix'''
        
        if n_filter:
            pass_cards = [card for card in self.card_data if self.card_data[card]['main'] > n_filter]
        else:
            pass_cards = None
            
        matrix = create_cooccurence_matrix(self.decks, drop_basics=drop_basics, card_subset=pass_cards)

        self.cooccurence_matrix = matrix
        return matrix