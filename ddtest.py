import dedupe
import os
import csv
import re
from pprint import pprint

input_file = 'Samples\\clean_sample.csv'
output_file = 'Output\\sampleoutput.csv'

settings_file = 'csv_example_learned_settings'
training_file = 'csv_example_training.json'


def preProcess(column):
    '''
    Removes quotes, BOM and whitespaces from the start and the end of records
    and makes everything lowercase for better comparison.
    '''
    column = column.strip().strip('"').strip("'").lower().strip()
    # Dedupe won't allow empty fields so we have to substitute it with None
    # we also take out na and NA because they basically mean the same
    if not column or column.lower() == 'na':
        return None
    else:
        return column

def readData(filename):
    '''
    reads a CSV file and returns a dictionary of dictionaries where keys are the value in the column set as id_field
    and the value is a row from the file represented as a dictionary.

    filename -> a valid path to a csv file - assuming it exists
        '''
    data_d = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # we need a unique record id which either should be a column in the file
        # or as in this case I just use the row number
        for i, row in enumerate(reader):
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            data_d[i] = dict(clean_row)

    return data_d


print('importing data ...')
data_d = readData(input_file)

# Checking for settings file
if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f) # if we use already saved settings we use the Static class
else:
    # define which fields are "important" and what data types they are
    fields = [
        {'field': 'name', 'type': 'String'},
        {'field': 'address', 'type': 'String'}, # there's an optional Address data type bu that's an additional package
        {'field': 'city', 'type': 'String'},
        {'field': 'cuisine', 'type': 'String'},
    ]

    # this is where magic begins
    deduper = dedupe.Dedupe(fields)
    deduper.sample(data_d, 15000)

    # checking for training file
    if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.readTraining(f)

    # Learning
    print('starting active labeling...')

    # this is where we "answer the questions"
    dedupe.consoleLabel(deduper)

    deduper.train()

    with open(training_file, 'w') as tf:
        deduper.writeTraining(tf)

    with open(settings_file, 'wb') as sf:
        deduper.writeSettings(sf)


# Find the threshold that will maximize a weighted average of our precision and recall.
# When we set the recall weight to 2, we are saying we care twice as much about recall as we do precision.
# https://en.wikipedia.org/wiki/Precision_and_recall
threshold = deduper.threshold(data_d, recall_weight=1.5)
print("threshold value: ", threshold)

# Clustering
print('clustering...')
clustered_dupes = deduper.match(data_d, threshold)

print('# duplicate sets', len(clustered_dupes))

# Writing results
cluster_membership = {}
cluster_id = 0
for (cluster_id, cluster) in enumerate(clustered_dupes):
    id_set, scores = cluster
    cluster_d = [data_d[c] for c in id_set]
    canonical_rep = dedupe.canonicalize(cluster_d)
    for record_id, score in zip(id_set, scores):
        cluster_membership[record_id] = {
            "cluster id": cluster_id,
            "canonical representation": canonical_rep,
            "confidence": score
        }

singleton_id = cluster_id + 1 # records that are not in a cluster

with open(output_file, 'w', encoding='utf-8') as f_output, open(input_file, encoding='utf-8') as f_input:
    writer = csv.writer(f_output)
    reader = csv.reader(f_input)

    heading_row = next(reader)
    heading_row.insert(0, 'confidence_score')
    heading_row.insert(0, 'Cluster ID')
    canonical_keys = canonical_rep.keys()
    for key in canonical_keys:
        heading_row.append('canonical_' + key)

    writer.writerow(heading_row)

    for i, row in enumerate(reader):
        row_id = i
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]["cluster id"]
            canonical_rep = cluster_membership[row_id]["canonical representation"]
            row.insert(0, cluster_membership[row_id]['confidence'])
            row.insert(0, cluster_id)
            for key in canonical_keys:
                row.append(canonical_rep[key].encode('utf8'))
        else:
            row.insert(0, 0) # confidence score 0 because we're sure it isn't a duplicate
            row.insert(0, singleton_id) # not in cluster
            singleton_id += 1

        writer.writerow(row)

    print('writing to output file...')
