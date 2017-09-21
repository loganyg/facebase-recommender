# September 6, 2017
# Logan Young
#
# A basic RESTful API on a flask server to access user and item recommendations
# for recommenders stores as pickle files.
#
# NOTE: This folder contains the basic flask code for the app. It does not
# include the files that are on the webserver (api.wsgi and index.html) that
# are necessary for the app to be deployed.
# The code here is for reference, actual changes must be added to the app on
# the server.

from flask import Flask, jsonify, make_response, abort
import pickle as pkl
from .fbrecommender.recommenders import *

app = Flask(__name__)

with open("./ui_recommender.pkl", 'rb') as pickled_r:
    uir = pkl.load(pickled_r)

with open("./ii_recommender.pkl", 'rb') as pickled_r:
    iir = pkl.load(pickled_r)


@app.route('/users/')
def users():
    ''' Retrieve the list of every user in the recommender.'''
    return jsonify({'users': uir.users.tolist()})


@app.route('/items/')
def items():
    ''' Retrieve the list of every item in the recommender'''
    return jsonify({'items': iir.items.tolist()})


@app.route('/users/<string:user_id>/recommendations/<int:rec_num>')
def user_recs(user_id, rec_num):
    '''
    Retrieve the top recommended items for a specified user.

    :param user_id: The unique identifier of the user for which to get
    recommendations.
    :param rec_num: The number of recommendations to retrieve.
    :return: The items in JSON format.
    '''
    if user_id not in uir.users:
        abort(404)
    if rec_num < 0:
        abort(400)
    return jsonify({
        'user': user_id,
        'recommendations':
        uir.recommendations.loc[user_id, :(rec_num - 1)].tolist()
    })


@app.route('/users/<string:user_id>/recommendations/start/<int:start>/end/<int:end>')
def user_recs_range(user_id, start, end):
    '''
    Retrieve a range of recommendations for a specified user.

    :param user_id: The unique identifier of the user for which to get
    recommendations.
    :param start: The beginning index for the range of recommendations.
    :param end: The terminal index for the range of recommendations.
    '''
    if user_id not in uir.users:
        abort(404)
    if start < 0 or start >= len(uir.items) or end < 0:
        abort(400)
    return jsonify({
        'user': user_id,
        'recommendations': uir.recommendations.loc[user_id, start:end].tolist()
    })


@app.route('/items/<string:item_id>/recommendations/<int:rec_num>')
def item_recs(item_id, rec_num):
    '''
    Retrieve the top recommended items for a specified item.

    :param item_id: The unique identifier of the item for which to get
    recommendations.
    :param rec_num: The number of recommendations to retrieve.
    :return: The items in JSON format.
    '''
    if item_id not in iir.items:
        abort(404)
    if rec_num < 0:
        abort(400)
    return jsonify({
        'item': item_id,
        'recommendations':
        iir.recommendations.loc[item_id, :(rec_num - 1)].tolist()
    })


@app.route('/items/<string:item_id>/recommendations/start/<int:start>/end/<int:end>')
def item_recs_range(item_id, start, end):
    '''
    Retrieve a range of recommendations for a specified item.

    :param item_id: The unique identifier of the item for which to get
    recommendations.
    :param start: The beginning index for the range of recommendations.
    :param end: The terminal index for the range of recommendations.
    '''
    if item_id not in uir.items:
        abort(404)
    if start < 0 or start >= len(iir.items) or end < 0:
        abort(400)
    return jsonify({
        'item': item_id,
        'recommendations':
        iir.recommendations.loc[item_id, start:end].tolist()
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not Found'}), 404)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad Request'}), 400)


if __name__ == '__main__':
    app.run()
