# September 18, 2017
# Logan Young
#
# Implementation of Collaborative Deep Learning (CDL) by Wang Hao et al.
# using keras with tensorflow backend.

from keras.layers import Input, Dense
from keras.layers.noise import GaussianNoise
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.models import Model
from keras import metrics
import numpy as np
import pandas as pd
import json
from numpy.linalg import inv
from .abstractrecommender import FBRecommender


class StackedDenoisingAutoencoder:
    '''
    Builds a stacked denoising autoencoder using keras. There are
    two outputs, the encoding and the decoding.

    :param layers: The number of layers for the SDAE, it must be a
    multiple of 2.
    :param content_num: The number of content features for the input.
    :param hp_w: The hyperparameter for the weights.
    :param hp_n: The hyperparameter for the decoded output.
    :param intermediate_units: The number of units for the hidden layers
    of the SDAE.
    :param factors: The number of latent factors for the encoding.
    '''

    def __init__(self,
                 content_num,
                 **kwargs):
        layers = kwargs.get('layers', 4)
        intermediate_units = kwargs.get('intermediate_units', 200)
        activation = kwargs.get('activation', 'sigmoid')
        hp_w = kwargs.get('hp_w', 10)
        hp_n = kwargs.get('hp_n', 100)
        factors = kwargs.get('factors', 20)
        inputs = Input(shape=(content_num,))
        # Add the encoding layers
        encoder = inputs
        # The first few encoding layers have the intermediary size K_l
        for i in range(1, int(layers / 2)):
            encoder = Dense(intermediate_units,
                            activation=activation,
                            kernel_initializer=RandomNormal(
                                mean=0.0,
                                stddev=1 / hp_w
                            ),
                            use_bias=True,
                            bias_initializer=RandomNormal(
                                mean=0.0,
                                stddev=1 / hp_w
                            ),
                            kernel_regularizer=l2(hp_w / 2),
                            bias_regularizer=l2(hp_w / 2)
                            )(encoder)
        # Add the final encoding layer, with a size equal to the number of
        # latent factors for the user and item vectors, so it can be piped
        # into the item factors.
        encoder = Dense(factors,
                        activation=activation,
                        kernel_initializer=RandomNormal(
                            mean=0.0,
                            stddev=1 / hp_w
                        ),
                        use_bias=True,
                        bias_initializer=RandomNormal(
                            mean=0.0,
                            stddev=1 / hp_w
                        ),
                        kernel_regularizer=l2(hp_w / 2),
                        bias_regularizer=l2(hp_w / 2),
                        name='encoder'
                        )(encoder)

        # Add decoder layers
        decoder = encoder
        # Add the inbetween decoder layers, with intermediary size K_l
        for i in range(0, int(layers / 2)):
            decoder = Dense(intermediate_units,
                            activation=activation,
                            kernel_initializer=RandomNormal(
                                mean=0.0,
                                stddev=1 / hp_w
                            ),
                            use_bias=True,
                            bias_initializer=RandomNormal(
                                mean=0.0,
                                stddev=1 / hp_w
                            ),
                            kernel_regularizer=l2(hp_w / 2),
                            bias_regularizer=l2(hp_w / 2)
                            )(decoder)
        # Add the final decoder layer, with a size of the clean data, equal
        # to the size of the lexicon.
        decoder = Dense(content_num,
                        activation=activation,
                        kernel_initializer=RandomNormal(
                            mean=0.0,
                            stddev=1 / hp_w
                        ),
                        use_bias=True,
                        bias_initializer=RandomNormal(
                            mean=0.0,
                            stddev=1 / hp_w
                        ),
                        kernel_regularizer=l2(hp_w / 2),
                        bias_regularizer=l2(hp_w / 2)
                        )(decoder)
        # Add a small amount of variance to the output
        decoder = GaussianNoise(1 / hp_n, name='decoder')(decoder)
        self.model = Model(inputs=inputs, outputs=[decoder, encoder])


class CDLUIRecommender(FBRecommender):
    '''
    Implementation of Hao Wang et al.'s Collaborative Deep Filtering using
    keras and numpy arrays.

    :param ui_datasource: The location of the user-item data, either as
    a .csv or .pkl filepath or a pandas DataFrame.
    :param ic_datasource: The location of the item-content data, as
    the filepath of a json file that holds a dictionary mapping items to
    a list of keywords.
    :param factors: The number of factors for the matrix factorization of
    the users and items.
    :param intermediate_units: The number of units for the hidden layers
    of the SDAE.
    :param layers: The number of layers in the SDAE, must be a multiple of
    2.
    :param noise_factor: The proportion of content data to add noise to
    to create the input for the SDAE.
    :param hp_w: The weights hyperparameter.
    :param hp_n: The SDAE decoded output hyperparameter.
    :param hp_v: The item hyperparameter.
    :param hp_u: The user hyperparameter.
    :param transpose_ui: Whether or not to transpose the user-item data
    after it has been loaded. This should be True if the user-item data
    is of the form <users> x <items>.
    :param pos_conf: The confidence value for positive (1) content data.
    :param neg_conf: The confidence value for negative (0) content data
    '''

    def __init__(self,
                 ui_datasource,
                 ic_datasource,
                 **kwargs):
        # Initializes user-item data
        self.hp_w = kwargs.get('hp_w', 10)
        self.hp_n = kwargs.get('hp_n', 100)
        self.hp_v = kwargs.get('hp_v', 10)
        self.hp_u = kwargs.get('hp_u', .1)
        self.factors = kwargs.get('factors', 20)
        self.hp_u_id = self.hp_u * np.identity(self.factors)
        self.hp_v_id = self.hp_v * np.identity(self.factors)
        transpose_ui = kwargs.get('transpose_ui', True)
        if type(ui_datasource) == tuple:
            self.ui_data = ui_datasource[0]
            self.users = ui_datasource[1]
            self.items = ui_datasource[2]
            if transpose_ui:
                self.ui_data = np.transpose(self.ui_data)
        else:
            if type(ui_datasource) == str:
                if ui_datasource.endswith(".csv"):
                    ui_data_pd = pd.read_csv(ui_datasource, index_col=0)
                elif ui_datasource.endswith(".pkl"):
                    ui_data_pd = pd.read_pickle(ui_datasource)
            elif type(ui_datasource) == pd.DataFrame:
                    ui_data_pd = ui_datasource
            if transpose_ui:
                ui_data_pd = ui_data_pd.transpose()
            self.items = ui_data_pd.index
            self.users = ui_data_pd.columns
            self.ui_data = ui_data_pd.as_matrix()
        # Create the confidence values matrix.
        conf_type = kwargs.get('conf_type', 'uniform')
        conf_data = np.copy(self.ui_data).astype(float)
        if conf_type == 'uniform':
            conf_data[conf_data != 1] = kwargs.get('pos_conf', 1)
            conf_data[conf_data == 0] = kwargs.get('neg_conf', .01)
        elif conf_type == 'scale':
            conf_data = (conf_data * 'conf_param') + 1
        self.conf_data = conf_data
        # Initialize item-content data
        if type(ic_datasource) == str:
            if(ic_datasource.endswith(".json")):
                with open(ic_datasource, 'r') as ic_text:
                    ic_data_json = json.load(ic_text)
                # Build item-content matrix
                # First Pass
                contentset = set()
                for item in self.items:
                    if item in ic_data_json.keys():
                        for word in ic_data_json[item]:
                            if word not in contentset:
                                contentset.add(word)
                self.content = pd.Index(contentset)
                self.ic_data = np.empty((len(self.items), len(self.content)))
                # Second Pass
                item_map = self.items.to_series()
                item_map[:] = range(0, len(item_map))
                content_map = self.content.to_series()
                content_map[:] = range(0, len(content_map))
                for item in self.items:
                    item_ix = item_map[item]
                    if item in ic_data_json.keys():
                        for word in ic_data_json[item]:
                            word_ix = content_map[word]
                            self.ic_data[item_ix, word_ix] = 1
        # Initialize similarity scores matrix
        self.sim_scores = np.empty((len(self.items), len(self.users)))
        # Add noise to the clean content data
        noised_data = np.random.uniform(
            size=(
                len(self.items),
                len(self.content)
            )
        )
        noised_data[noised_data < kwargs.get('noise_factor', .3)] = 2
        noised_data[noised_data != 2] = 0
        noised_data = noised_data + self.ic_data
        noised_data[noised_data == 3] = 0
        noised_data[noised_data == 2] = 1
        self.noised_data = noised_data
        # Initialize the Stacked Denoising Autoencoder (SDAE)
        self.sdae = StackedDenoisingAutoencoder(
            len(self.content),
            **kwargs)
        self.sdae.model.compile(optimizer='sgd',
                                loss='mse',
                                loss_weights={
                                     'encoder': self.hp_v / 2,
                                     'decoder': self.hp_n / 2},
                                metrics=[metrics.binary_accuracy, ])
        # Initialize the user and item factors
        self.u_factors = np.random.normal(
            loc=0.0,
            scale=1 / self.hp_u,
            size=(len(self.users), self.factors)
        )
        self.i_factors = self.sdae.model.predict(self.ic_data)[1]
        + np.random.normal(
            loc=0.0,
            scale=1 / self.hp_v,
            size=(len(self.items), self.fasctors)
        )
        super(CDLUIRecommender, self).__init__()

    def update_u_factors(self):
        '''
        Assume the item factors are fixed and update the user factors
        based on that assumption.
        '''
        for u in range(0, len(self.users)):
            conf_mult_matrix = np.tile(
                self.conf_data[:, u], (self.factors, 1))
            self.u_factors[u, :] = np.dot(
                np.dot(
                    inv(
                        np.dot(
                            self.i_factors.T * conf_mult_matrix,
                            self.i_factors) + self.hp_u_id
                    ),
                    self.i_factors.T) * conf_mult_matrix,
                self.ui_data[:, u])

    def update_i_factors(self, encoding):
        '''
        Assume the user factors and item encoding is fixed and update
        the item factors based on the assumption.

        :param encoding: The <item_num>x<factor_num> matrix that stores
        the most recent encoding of the item-content data by the SDAE.
        '''
        for i in range(0, len(self.items)):
            conf_mult_matrix = np.tile(
                self.conf_data[i, :], (self.factors, 1))
            self.i_factors[i, :] = np.dot(
                inv(
                    np.dot(
                        self.u_factors.T * conf_mult_matrix,
                        self.u_factors
                    ) + self.hp_v_id
                ),
                np.dot(
                    self.u_factors.T * conf_mult_matrix,
                    self.ui_data[i, :]
                ) + (self.hp_v * encoding[i, :])
            )

    def update_factors(self, iterations, encoding):
        '''
        Alternate between updating the user and item factors.

        :param iterations: The number of times to update both
        factor matrices.
        :param encoding: The <item_num>x<factor_num> matrix that stores
        the most recent encoding of the item-content data by the SDAE.
        '''
        for _ in range(0, iterations):
            self.update_u_factors()
            self.update_i_factors(encoding)

    def populate_recommender(self, **kwargs):
        '''
        Run the algorithm to train the recommender on the data provided
        to the intializer. The CDL method alternates between fixing the SDAE
        and the user and item factors to optimize the opposing part.

        :param iterations: The number of times to run the optimization.
        :param verbose: How much information to print out durnig training, 0-3.
        '''
        iterations = kwargs.get('iterations', 100)
        verbose = kwargs.get('verbose', 1)
        for i in range(0, iterations):
            if verbose >= 1:
                print('Actual Epoch %s/%s' % (i + 1, iterations))
            model_verbose = max(0, verbose - 1)
            self.sdae.model.fit(self.noised_data,
                                {
                                    'decoder': self.ic_data,
                                    'encoder': self.i_factors
                                },
                                epochs=1,
                                batch_size=32,
                                verbose=model_verbose
                                )
            encoding = self.sdae.model.predict(self.ic_data)[1]
            self.update_factors(1, encoding)
        for u in range(0, len(self.users)):
            for i in range(0, len(self.items)):
                if self.ui_data[i, u] != 0:
                    self.sim_scores[i, u] = 0
                else:
                    self.sim_scores[i, u] = np.dot(
                        self.u_factors[u, :], self.i_factors[i, :])
        self.sim_scores_sorted = np.sort(self.sim_scores, axis=0)
        self.sim_scores_argsorted = np.argsort(self.sim_scores, axis=0)
        self.sim_scores = np.transpose(self.sim_scores)
        self.sim_scores_sorted = np.transpose(self.sim_scores_sorted)
        self.sim_scores_argsorted = np.transpose(self.sim_scores_argsorted)
        for u in range(0, len(self.users)):
            self.recommendations.ix[self.users[u], :] = self.items[
                np.flip(self.sim_scores_argsorted[u], 0)
            ]
