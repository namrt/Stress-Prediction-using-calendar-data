from sklearn.feature_extraction.text import CountVectorizer
from classifier import training, testing
from sklearn import preprocessing
from datetime import datetime
from numpy import savetxt
import collections
import numpy as np
import pandas as pd
import pickle 
import os
import csv
import math
import sys
import iso8601

def key_words_builder(key):
    keywords = []
    folder = "keywords/"
    with open(folder+key+".txt") as f:
        for line in f:
            temp_list = line.strip().split("\n")
            keywords += [x for x in temp_list]
    return keywords

def company_email(attendees, email_domain):
    if not(pd.notnull(attendees)):
        return 0
    attendees = attendees.split("; ")   
    ctr = 0
    for a in attendees:
        if a.endswith(email_domain):
            ctr +=1
    if ctr > 1: 
        return  1 
    else:
        return 0 

def is_online(row):
     row  = row.lower()
     if "zoom" in row:
         return 1 
     else:
         return 0 

def participant(row):
    row = row
    new_row = []
    for e in row:
        e = e.lower()
        if e in emails_ids.keys():
            new_row += [emails_ids[e]]
    return new_row

def morning(input):
    input = datetime.strptime(input,"%Y-%m-%dT%H:%M:%S%z")
    if 5<=input.hour < 12:
        return 1
    else:
        return 0

def process_time(input):
    date_with_tz =input 
    dt = datetime.strptime(input,"%Y-%m-%dT%H:%M:%S%z")
    value = str(dt)
    dt = iso8601.parse_date(value)
    dt = datetime.timestamp(dt) 
    return dt


def filter(name, keywords, data):
    vectorizer = CountVectorizer(vocabulary = set(keywords))
    data_vectorized = vectorizer.fit_transform(data["summary"])
    data_np = data_vectorized.toarray() > 0
    name = name+"-key"
    data[name] = np.sum(data_np, axis = 1, keepdims = True)
    data[name] = data[name].apply(lambda x: 1.0*(x>0))
    return data


def process(file_name, phase,time_stamp_inf_file_name):
    # Load the Pandas libraries with alias 'pd'
    data = pd.read_csv(file_name)

    influences = ["managers", "co-workers", "customers", 
                  "meetings", "tasks", "thoughts", "finances", 
                  "health", "spouse-partner", "children",
                  "parents","friends", "body"]

    for value in influences: 
        data = filter(value, key_words_builder(value), data)


    ### is online 
    data["is_online"] = 0
    data["is_online"] = data["location"].apply(lambda x: is_online(x) if pd.notnull(x) else 0)


    ### coworkers
    data["co-workers-key"] = 1*(data.apply(lambda x: company_email(x["attendees"],x["email_domain_job"]), axis=1) |  (data["co-workers-key"] ==1))

    ### meeting and coworkers
    data["meetings-key"] =  1*((data["meetings-key"] ==1) |  (data["co-workers-key"] ==1))
    data.rename(columns={"spouse-partner-key": "spouse/partner-key"}, inplace=True)

    ### redefine the label to replace "spouse-partner"
    influences = ["managers", "co-workers", "customers", 
                  "meetings", "tasks", "thoughts", "finances", 
                  "health", "spouse/partner", "children",
                  "parents","friends", "body", "other"]

    influences_key = [x+"-key" for x in influences] 
    data["other-key"] = 0
    data["other-key"] = data[influences_key].sum(axis = 1, skipna = True)
    data["other-key"] =  data["other-key"].apply(lambda x: 0 if x>0 else 1) 
    
    ### timestamp
    data["start_timestamp"] = data["start_date_time"].apply(lambda x: process_time(x)) 
    data["end_timestamp"] = data["end_date_time"].apply(lambda x: process_time(x)) 

    data.dropna(subset = ["start_timestamp"], inplace=True)
    
    if (phase == "training"):
        min_v_start = data["start_timestamp"].dropna().min()
        max_v_start = data["start_timestamp"].dropna().max()
        min_v_end = data["end_timestamp"].dropna().min()
        max_v_end = data["end_timestamp"].dropna().max()
    elif  phase == "testing":
        # normalize the start_timestamp
        time_stamp_info = pickle.load(open(time_stamp_info_file_name, 'rb')) 
        min_v_start = time_stamp_info["min_v_start"] 
        max_v_start = time_stamp_info["max_v_start"] 
        min_v_end = time_stamp_info["min_v_end"] 
        max_v_end = time_stamp_info["max_v_end"] 
    data["angle_start"] = data["start_timestamp"].apply(lambda x: math.acos((x - min_v_start) / (max_v_start- min_v_start)))
    data["r_start"] = data["start_timestamp"].apply(lambda x: x/max_v_start)


    #morning? 
    data["morning"] = data["start_date_time"].apply(lambda x: morning(x)) 

    # normalize the end_timestamp
    data["angle_end"] = data["end_timestamp"].apply(lambda x: math.acos((x - min_v_end)/(max_v_end- min_v_end)))
    data["r_end"] = data["end_timestamp"].apply(lambda x: x/max_v_end)
    data["duration"] = (data["end_timestamp"] - data["start_timestamp"])/60
    data["attendees"] = data["attendees"].astype(str)
    data["attendees"] = data["attendees"].apply(lambda x: x.split("; "))
    data["num_participants"] = data["attendees"].apply(lambda x: len(x))

    # multi-class one-hot transformation for participants
    #data["participants"] = data["attendees"].apply(lambda x: participant(x))
    #mlb = preprocessing.MultiLabelBinarizer()
    #mlb.fit([list(bio_dict.keys())])
    #data = data.join(pd.DataFrame(mlb.transform(data.pop("participants")), 
    #                 columns=mlb.classes_, index=data.index))

    #data["creatorOrNot"] = 1*(data["creator"].map(emails_ids) == data["user_id"])

    features = influences_key + ["event_id", "user_id","duration", "is_online","angle_start", "num_participants",
                                 "r_start", "angle_end","r_end", "duration", "morning"]

    influence_cols = ["influence1", "influence2", "influence3"]
    categories = ["stress_level"] 
    if phase == "training":
        data = data[features+ categories]
        data.dropna(inplace=True)
        time_stamp_info = {"max_v_end":max_v_end, "min_v_end":min_v_end,
                            "max_v_start":max_v_start,"min_v_start":min_v_start}
        pickle.dump(time_stamp_info, open(time_stamp_info_file_name, 'wb'))
        return data,categories 
    if phase == "testing":
        data = data[features]
        return data,None

if __name__ == "__main__":
    # training
    # time_stamp_info_file_name keeps the paramter for normalizing the time-stamp
    time_stamp_info_file_name = "pickle/time_stamp_info.p"
    # process the data
    train_data,labels = process("input/train.csv", "training",time_stamp_info_file_name)
    # train the model
    pca_file_name = "pickle/pca.p"
    trained_file_name = "pickle/trained_model.p"
    training(train_data, pca_file_name, trained_file_name)
    # testing
    # process the data
    test_data,_ = process("input/test.csv", "testing", time_stamp_info_file_name)
    #test the model
    testing(test_data, pca_file_name, trained_file_name)

