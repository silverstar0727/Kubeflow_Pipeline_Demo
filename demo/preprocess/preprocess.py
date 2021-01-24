import sys
import pandas as pd
import argparse

def preprocessing(raw_csv_gcs_uri, preproc_csv_gcs_uri):
    print('Preprocessing Titanic Data')
    train_raw = pd.read_csv(raw_csv_gcs_uri)

    features = ['Pclass', 'Sex', 'SibSp']
    preprocessing_data = pd.get_dummies(train_raw[features])
    preprocessing_data['Survived'] = train_raw['Survived']

    print(preprocessing_data.head())
    preprocessing_data.to_csv(preproc_csv_gcs_uri, index = False)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--raw_csv_gcs_uri',
        type=str,
        required=True,
        help='Google Cloud Storage URI for reading raw train data (ex: gs://{bucketName}/{pathToRaw}/{rawTrainFile}.csv)'
    )

    arg_parser.add_argument(
        '--preproc_csv_gcs_uri',
        type=str,
        required=True,
        help='Google Cloud Storage URI for writing pre-processed train data (ex: gs://{bucketName}/{pathToRaw}/{rawTrainFile}.csv)'
    )

    args, unknown = arg_parser.parse_known_args()
    preprocessing(args.raw_csv_gcs_uri, args.preproc_csv_gcs_uri)