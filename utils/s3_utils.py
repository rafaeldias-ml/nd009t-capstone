import boto3
import pandas as pd

class BucketUtils:
    """
    Utility class to hold bucket related information and to provide utility methods
    to read and save csv and parquet files to s3 bucket
    """
    def __init__(self, bucket_name, prefix):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)

    def get_files(self):
        s3_files = set([object_summary.key for object_summary in self.bucket.objects.filter(Prefix=self.prefix)])

        return s3_files

    def read_csv(self, filename, **kwargs):
        data_location = 's3://{}/{}/{}'.format(self.bucket_name, self.prefix, filename)

        return pd.read_csv(data_location, **kwargs)

    def save_csv(self, df, filename, **kwargs):
        data_location = 's3://{}/{}/{}'.format(self.bucket_name, self.prefix, filename)

        return df.to_csv(data_location, **kwargs)

    def read_parquet(self, filename, **kwargs):
        data_location = 's3://{}/{}/{}'.format(self.bucket_name, self.prefix, filename)

        return pd.read_parquet(data_location, **kwargs)

    def save_parquet(self, df, filename, **kwargs):
        data_location = 's3://{}/{}/{}'.format(self.bucket_name, self.prefix, filename)

        return df.to_parquet(data_location, **kwargs)

    def exists(self, filename):
        s3_files = self.get_files()

        return '{}/{}'.format(self.prefix, filename) in s3_files
