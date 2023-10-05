import math
import heapq
import multiprocessing
import numpy as np
from time import time


_model = None
_testRatings = None
_testNegatives = None
_K = None
_test_full_user = None
_flatten_test_sum_Ratings = None
_test_sum_Ratings = None

def evaluate_model(model, testRatings, test_full_user,flatten_test_sum_Ratings, test_sum_Ratings, K):
    
    global _model
    global _testRatings
    global _K

    #global _val_full_user
    global _test_full_user
    #global _val_sum_Ratings 
    global _flatten_test_sum_Ratings
    #global _val_item_pred_dict 
    global _test_sum_Ratings

    #_val_full_user = val_full_user
    _test_full_user = test_full_user
    #_val_sum_Ratings = val_sum_Ratings
    _flatten_test_sum_Ratings = flatten_test_sum_Ratings
    #_val_item_pred_dict = val_item_pred_dict
    _test_sum_Ratings = test_sum_Ratings

    _model = model
    _testRatings = testRatings
    _K = K    

    hits, ndcgs = [],[]

    items = _flatten_test_sum_Ratings
    users = _test_full_user
    predictions = _model.predict([np.array(users), np.array(items)], 
                                 batch_size=2**1000, verbose=0)

    predictions[100:200][0]
    for n,i in enumerate(_test_sum_Ratings):
        rating = _testRatings[n]
        gtItem = rating[1]


        val_map_item_score = {
            j: predictions[100 * n : 100 * n + 100][m][0]
            for m, j in enumerate(i)
        }
        ranklist = heapq.nlargest(_K, val_map_item_score, key=val_map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)

        hits.append(hr)
        ndcgs.append(ndcg)

    return (hits, ndcgs)        
        

def getHitRatio(ranklist, gtItem):
    return next((1 for item in ranklist if item == gtItem), 0)

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
