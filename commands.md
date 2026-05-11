

# pull (delete local files not in S3)
aws s3 sync s3://forecasting-nem-dd . --region ap-southeast-2 --exclude ".git/*" --exclude "2_Features_build/Feature_data/*" --delete


# push (delete s3 files not in local)
aws s3 sync . s3://forecasting-nem-dd --region ap-southeast-2 --exclude ".git/*" --exclude "2_Features_build/Feature_data/*" --delete
