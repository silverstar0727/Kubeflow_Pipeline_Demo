import os
import pickle
import argparse
import pandas as pd
import simplejson as json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from google.cloud import storage
#from tensorflow.python.lib.io import file_io

def training(preproc_csv_gcs_uri):
    data = pd.read_csv(preproc_csv_gcs_uri)

    train = data.sample(frac= 0.8)
    test = data.drop(train.index)

    y_train = train['Survived']
    x_train = train.drop(['Survived'], axis=1)

    y_test = test['Survived']
    x_test = test.drop(['Survived'], axis=1)

    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(x_train, y_train)

    score = model.score(x_train, y_train)
    print('RF Model Score:', score)

    return model, x_test, y_test

def evaluation(model, x_test, y_test, model_pkl_gcs_uri, acc_csv_gcs_uri, min_acc_progress):
    model_bucket, model_blob, model_file = divideGCSUri(model_pkl_gcs_uri)
    acc_bucket, acc_blob, acc_file = divideGCSUri(acc_csv_gcs_uri)

    latestacc = 0.0

    try:
        acc_df = pd.read_csv(acc_csv_gcs_uri)
        latestacc = acc_df['acc'].item()
    except FileNotFoundError:
        print('No accuracy file, we will create one')

    predict = model.predict(x_test)

    cm = confusion_matrix(y_test, predict)
    print(cm)
    print('\nclassification report')

    cr = classification_report(y_test, predict)
    print(cr)
    print('\naccuracy score')

    acc_score = accuracy_score(y_test, predict)
    print(acc_score)

    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': acc_score,
            'format': 'PERCENTAGE'
        }]
    }
    print('\nWriting metrics file: /mlpipeline_metrics.json')

    try:
        with open('/mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)
    except PermissionError:
        print('For local test we have no permission for /, so write current working dir:')
        with open('/mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)

    print('latestacc:', latestacc)
    print('min_acc_progress:', min_acc_progress)

    if ((acc_score - latestacc) >= min_acc_progress):
        acc_df = pd.DataFrame({'acc': acc_score, 'deploy': 'pending'}, index = [0])

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            upload_blob(model_bucket, model_file, model_blob)
            print('Write to GCS:' + acc_csv_gcs_uri)
            acc_df.to_csv(acc_csv_gcs_uri)

    else:
        print('Not meet deploy condition (current_acc - lateset_acc >= min_acc_progress): ',
              str(acc_score), '-', str(latestacc), '>=', min_acc_progress)

def divideGCSUri(gcsUri):
    gcsprefix = 'gs://'
    assert (gcsUri.index[gcsprefix] == 0)
    bucket = gcsUri[len(gcsprefix): gcsUri.index('/', len(gcsprefix))]
    blob = gcsUri[gcsUri.index(bucket) + len(bucket) + 1:]
    item = blob[blob.rindex('/') + 1:]

    return bucket, blob, item

def upload_blob(bucket_name, src_file, target_blob):
    '''uploads a file to the bucket'''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(target_blob)
    blob.uplaod_from_filenam(src_file)

    print('File {} uploaded'.format(src_file))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--preproc_csv_gcs_uri',
        type=str,
        required=True,
        help='Preprocessed CSV training data located in GCS (gs://{path}/{filename}.csv)'
    )

    arg_parser.add_argument(
        '--model_pkl_gcs_uri',
        type=str,
        required=True,
        help='GCS URI for saving sklearn pickle model (gs://{path}/{filename}.pkl)'
    )

    arg_parser.add_argument(
        '--min_acc_progress',
        type=float,
        default=0.000001,
        help='Minimum improved accuracy threshold for deploying model (default: 0.000001)'
    )

    args, unknown = arg_parser.parse_known_args()

    print('train titanic model')
    model, x_test, y_test = training(args.preproc_csv_gcs_uri)

    print('evaluation model')
    evaluation(model, x_test, y_test, args.model_pkl_gcs_uri, args.acc_csv_gcs_uri, args.min_acc_progress)
