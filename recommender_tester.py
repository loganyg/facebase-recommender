'''
August 23, 2017
Logan Young

A collection of functions to test various types of recommendation
engines. Used for both simple repeated testing/debugging of the engines
as well as cross-validation and/or grid search to find values of parameters
such as the regularization parameter and weighting parameter to optimize the
effectiveness of the engine for a set of data.
'''

import numpy as np
import pandas as pd
import re
from itertools import product


def run_tests(datasource,
              recommender,
              **kwargs) -> dict:
    '''Runs the specified number of tests for each permutations
    of the parameters. Parameters are tested by passing a tuple of
    values to test, while non-tuple arguments are passed through to
    the test of the model.

    :param datasource: The source of the user-item rating data.
    :param recommender: The FBRecommender class to use for making
    recommendations.
    :param verbose: How much supplemental information to print out, 0-3.
    :param metrics: A list of metrics to use to evaluate the tests of
    the recommender.
    :param reps: The number of tests to perform for each permutation of
    the paramters.
    '''
    test_params = ()
    test_values = ()
    for key in kwargs:
        if type(kwargs[key]) == tuple:
            test_params += (key,)
            test_values += (kwargs[key],)
    reps = kwargs.get('reps', 5)
    kwargs['metrics'] = kwargs.get(
        'metrics',
        ['rank', 'top_5_pct', 'top_10_pct']
    )
    metrics = kwargs['metrics']
    verbose = kwargs.get('verbose', 1)
    results = pd.DataFrame(columns=test_params)
    for metric in metrics:
        results["mean_%s" % metric] = None
        for i in range(0, reps):
            results["%s_%s" % (metric, i)] = None
    for values in product(*test_values):
        row_i = len(results)
        for i in range(0, len(test_params)):
            kwargs[test_params[i]] = values[i]
        results.loc[row_i] = values + (((0,) * (reps + 1)) * len(metrics))
        if verbose >= 1:
            print('\n' + ', '.join(['%s = %s' % (
                test_params[i], values[i]
            ) for i in range(
                0,
                len(test_values)
            )]))
        for rep in range(0, reps):
            if verbose >= 2:
                print('\nRep = %s' % rep)
            if kwargs.get('temporal', False):
                test_model_temporal(datasource,
                                    recommender,
                                    **kwargs)
            else:
                metric_vals = test_model(datasource,
                                         recommender,
                                         **kwargs)
            for metric in metrics:
                val = metric_vals[metric]
                if verbose >= 2:
                    print(
                        "Final %s = %s" % (
                            metric, val
                        )
                    )
                results.ix[
                    row_i,
                    '%s_%s' % (metric, rep)
                ] = val
                results.ix[row_i, 'mean_%s' % metric] += val
        for metric in metrics:
            results.ix[
                row_i,
                'mean_%s' % metric
            ] = results.ix[
                row_i,
                'mean_%s' % metric
            ] / float(reps)
            if verbose >= 1:
                print(
                    'Mean %s = %s' % (
                        metric,
                        results.ix[row_i, 'mean_%s' % metric]
                    )
                )
    if "write_to" in kwargs.keys():
        results.to_csv(kwargs['write_to'])
    return results


def calc_rank(sims_argsorted, testing_i, testing_j, full_data):
    n_users = sims_argsorted.shape[0]
    n_items = sims_argsorted.shape[1]
    rank_matrix = np.empty((n_users, n_items))
    for u in range(0, n_users):
        for i in range(0, n_items):
            rank_matrix[u, i] = (n_items - 1) - np.where(
                sims_argsorted[u] == i
            )[0][0]
    rank_matrix = rank_matrix / float(n_items - 1)
    rank = sum(
        full_data[
            testing_i,
            testing_j
        ] * rank_matrix[
            testing_i,
            testing_j
        ]
    ) / sum(full_data[testing_i, testing_j])
    return rank


def calc_top_n(n, sims_argsorted, testing_i, testing_j):
    n_users = sims_argsorted.shape[0]
    n_items = sims_argsorted.shape[1]
    top_matrix = np.zeros((n_users, n_items))
    for u in range(0, n_users):
        top_ix = sims_argsorted[u, (-1 * n):]
        top_matrix[(u,) * n, top_ix] = 1
    top_ratio = float(
        np.sum(
            top_matrix[testing_i, testing_j]
        )
    ) / float(len(testing_i))
    return top_ratio


def calc_top_in_n(n, sims_argsorted, testing_i, testing_j):
    n_users = sims_argsorted.shape[0]
    n_items = sims_argsorted.shape[1]
    test_matrix = np.zeros((n_users, n_items))
    test_matrix[testing_i, testing_j] = 1
    ratio_in = 0
    for u in range(0, n_users):
        top_ix = sims_argsorted[u, (-1 * n):]
        ratio_in += sum(test_matrix[u, top_ix])
    ratio_in = float(ratio_in) / float(n * n_users)
    return ratio_in


def calc_metrics(metrics, sims_argsorted, testing_i, testing_j, full_data):
    results = {}
    for metric in metrics:
        if re.match('top_[0-9]+', metric):
            threshold = int(re.findall("\d+", metric)[0])
            results[metric] = calc_top_n(
                threshold,
                sims_argsorted,
                testing_i,
                testing_j
            )
        if re.match('top_[0-9]+_pct', metric):
            threshold = int(re.search('[0-9]+', metric).group())
            results[metric] = calc_top_n_pct(
                threshold,
                sims_argsorted,
                testing_i,
                testing_j
            )
        if re.match('top_in_[0-9]+', metric):
            threshold = int(re.findall("\d+", metric)[0])
            results[metric] = calc_top_in_n(
                threshold,
                sims_argsorted,
                testing_i,
                testing_j
            )
        if metric == 'rank':
            results[metric] = calc_rank(
                sims_argsorted,
                testing_i,
                testing_j,
                full_data
            )
    return results


def test_model(datasource,
               recommender,
               **kwargs):
    '''
    Splits the data into a training and test set, then runs the specified model
    on the training set and runs various evaluative metrics on the test data.

    :param datasource: The source of the adjacency matrix inputted to the
    model. The input can be in either a DataFrame with users in the index
    and items in the columns or as a string file path to a csv.
    :param model: The type of model to use, either 'ALS' (Yifan Hu et al.) or
    'heuristic'.
    :param test_proportion: The proportion of the data to reserve for testing,
    while the rest will be inputted to train the model.
    :param metrics: A tuple of strings that store the names of each metric
    to evaluate this run of the model on.
    :param dump_to: The filepath to write the output as a csv, if not included
    the output will not be written.
    :param kwargs: The remaining arguments will be passed to the model.

    :returns: A dictionary mapping metric names to the values for this run of
    the model.
    '''
    test_proportion = kwargs.get('test_proportion', .2)
    metrics = kwargs.get('metrics', ('top_5_pct', 'rank'))
    if type(datasource) == pd.DataFrame:
        df = datasource
    elif type(datasource) == str:
        df = pd.read_csv(datasource, index_col=0)
    users = df.index
    items = df.columns
    full_data = df.as_matrix()
    training_data = np.copy(full_data)
    nz_i, nz_j = np.nonzero(full_data)
    ix = np.random.choice(
        len(nz_i),
        int(test_proportion * float(len(nz_i))),
        replace=False
    )
    training_data[nz_i[ix], nz_j[ix]] = 0
    r = recommender(
        (training_data, users, items),
        **kwargs
    )
    r.populate_recommender(**kwargs)
    if 'dump_model_to' in kwargs.keys():
        pd.DataFrame(
            r.sim_scores,
            index=users,
            columns=items
        ).to_csv(kwargs['dump_model_to'])
    return calc_metrics(metrics,
                        r.sim_scores_argsorted,
                        nz_i[ix],
                        nz_j[ix],
                        full_data
                        )


def calc_top_n_pct(n,
                   sims_argsorted,
                   testing_i,
                   testing_j):
    n_users = sims_argsorted.shape[0]
    n_items = sims_argsorted.shape[1]
    cutoff = int(n_items * (n / 100))
    top_matrix = np.zeros((n_users, n_items))
    for u in range(0, n_users):
        top_ix = sims_argsorted[u, (-1 * cutoff):]
        top_matrix[(u,) * cutoff, top_ix] = 1
    top_ratio = float(
        np.sum(
            top_matrix[testing_i, testing_j]
        )
    ) / float(len(testing_i))
    return top_ratio


def calc_next_n_in_m_pct(n,
                         m,
                         sims_argsorted,
                         test_actions,
                         users,
                         items):
    n_users = sims_argsorted.shape[0]
    n_items = sims_argsorted.shape[1]
    cutoff = int(n_items * (m / 100))
    top_matrix = np.zeros((n_users, n_items))
    for u in range(0, n_users):
        top_ix = sims_argsorted[u, (-1 * cutoff):]
        top_matrix[(u,) * cutoff, top_ix] = 1
    action_count = pd.Series(data=0, index=users.index)
    testing_i = []
    testing_j = []
    for i in test_actions.index:
        user = test_actions.loc[i, 'source']
        item = test_actions.loc[i, 'target']
        if action_count[user] < n:
            testing_i.append(users[user])
            testing_j.append(items[item])
            action_count[user] += 1
    top_ratio = float(
        np.sum(
            top_matrix[testing_i, testing_j]
        )
    ) / float(len(testing_i))
    return top_ratio


def calc_temporal_metrics(metrics,
                          sims_argsorted,
                          testing_i,
                          testing_j,
                          full_data,
                          test_actions,
                          users,
                          items):
    results = {}
    for metric in metrics:
        if re.match('next_[0-9]+_in_[0-9]+_pct', metric):
            thresholds = re.findall('[0-9]+', metric)
            results[metric] = calc_next_n_in_m_pct(
                int(thresholds[0]),
                int(thresholds[1]),
                sims_argsorted,
                test_actions,
                users,
                items
            )
        if re.match('top_[0-9]+_pct', metric):
            threshold = int(re.search('[0-9]+', metric).group())
            results[metric] = calc_top_n_pct(
                threshold,
                sims_argsorted,
                testing_i,
                testing_j
            )
        if metric == 'rank':
            results[metric] = calc_rank(
                sims_argsorted,
                testing_i,
                testing_j,
                full_data
            )
    return results


def test_model_temporal(datasource,
                        recommender,
                        **kwargs):
    '''
    Splits the data based on time instead of randomly, then tests
    the model on the time-split training and testing data.

    :param datasource: The source of the user-item ratings in the
    form of a timestamped edgelist.
    :param source_col: The name of source column for the edgelist.
    :param target_col: The name of target column for the edgelist.
    :param time_col: The name of the time column for the edgelist.
    :param weight_col: The name of the weights column for the edgelist.
    :param max_rating: The maximum rating allowed for the unit-item
    adjacency matrix.
    :param test_proportion: The portion of the data to save for testing.
    :param test_split_type: The method for splitting the test/training data.
    '''
    source_col = kwargs.get('source_col', 'source')
    target_col = kwargs.get('target_col', 'target')
    time_col = kwargs.get('time_col', 'time')
    weight_col = kwargs.get('weight_col', 'weight')
    max_rating = kwargs.get('max_rating', 5)
    test_proportion = kwargs.get('test_proportion', .2)
    test_split_type = kwargs.get('test_split_type', 'next_1')
    if type(datasource) == str:
            edgelist = pd.read_csv(
                datasource,
                dtype={source_col: object,
                       target_col: object}
            )
    elif type(datasource) == pd.DataFrame:
        edgelist = datasource
    # First pass to scrape users and items.
    itemset = set()
    userset = set()
    for i in edgelist.index:
        if edgelist.loc[i, source_col] not in userset:
            userset.add(edgelist.loc[i, source_col])
        if edgelist.loc[i, target_col] not in itemset:
            itemset.add(edgelist.loc[i, target_col])
    df = pd.DataFrame(index=userset, columns=itemset, data=0)
    user_map = df.index.to_series()
    user_map[:] = range(0, len(user_map))
    item_map = df.columns.to_series()
    item_map[:] = range(0, len(item_map))
    users = df.index
    items = df.columns
    timesorted_edgelist = edgelist.sort_values(time_col, axis=0)
    dupl_actions = []
    for i in timesorted_edgelist.index:
        user = timesorted_edgelist.loc[i, source_col]
        item = timesorted_edgelist.loc[i, target_col]
        weight = timesorted_edgelist.loc[i, weight_col]
        if df.loc[user, item] not in (weight, max_rating):
            df.loc[user, item] += weight
        else:
            dupl_actions.append(i)
    timesorted_edgelist.drop(dupl_actions, inplace=True)
    full_data = df.as_matrix()
    if test_split_type == 'chunk':
        cutoff = int(len(timesorted_edgelist.index) * (1 - test_proportion))
        cutoff_ix = timesorted_edgelist.index[cutoff]
        test_users = timesorted_edgelist.loc[cutoff_ix:, source_col]
        test_items = timesorted_edgelist.loc[cutoff_ix:, target_col]
        test_actions = timesorted_edgelist[cutoff_ix:]
    elif test_split_type.startswith('next_'):
        test_users = []
        test_items = []
        test_actions_ix = []
        n = int(re.findall("\d+", test_split_type)[0])
        actions_removed = pd.Series(index=range(0, len(users)), data=0)
        preferences = np.copy(full_data)
        preferences[preferences > 0] = 1
        user_sums = np.sum(preferences, axis=1)
        for i in range(0, len(user_sums)):
            if user_sums[i] <= n:
                actions_removed[i] = n
        flipped_tsel = timesorted_edgelist.sort_values(
            time_col, axis=0,
            ascending=False
        )
        for i in flipped_tsel.index:
            user = timesorted_edgelist.loc[i, source_col]
            item = timesorted_edgelist.loc[i, target_col]
            if actions_removed[user_map[user]] < n:
                test_users.append(user)
                test_items.append(item)
                test_actions_ix.append(i)
                actions_removed[user_map[user]] += 1
        test_actions = timesorted_edgelist.loc[test_actions_ix]
    training_data = np.copy(full_data)
    t_u_ix = user_map.loc[test_users]
    t_i_ix = item_map.loc[test_items]
    print(len(t_u_ix))
    training_data[t_u_ix, t_i_ix] = 0
    r = recommender(
        (training_data, users, items),
        **kwargs
    )
    r.populate_recommender(**kwargs)
    if 'write_to' in kwargs.keys():
        pd.DataFrame(
            r.sim_scores,
            index=users,
            columns=items
        ).to_csv(kwargs['write_to'])
    # Calculate different metrics
    return calc_temporal_metrics(kwargs.get('metrics', ['top_5_pct', ]),
                                 r.sim_scores_argsorted,
                                 t_u_ix,
                                 t_i_ix,
                                 full_data,
                                 test_actions,
                                 user_map,
                                 item_map
                                 )
