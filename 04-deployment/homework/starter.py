#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys


def read_data(filename, year, month, categorical):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df

def model_predict(df, dv, lr, categorical):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return lr.predict(X_val)


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    categorical = ['PUlocationID', 'DOlocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    print(input_file)

    df = read_data(
        input_file,
        year,
        month, 
        categorical
    )

    y_pred = model_predict(df, dv, lr, categorical)
    print(y_pred.mean())
    save_results(df, y_pred, 'df_result.parquet')


if __name__ == '__main__':
    run()
