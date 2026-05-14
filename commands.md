

# pull (delete local files not in S3)
aws s3 sync s3://forecasting-nem-dd . --region ap-southeast-2 --exclude ".git/*" --exclude "2_Features_build/Feature_data/*" --delete


# push (delete s3 files not in local)
aws s3 sync . s3://forecasting-nem-dd --region ap-southeast-2 --exclude ".git/*" --exclude "2_Features_build/Feature_data/*" --delete


pkill -9 -f ipykernel
pkill -9 -f lightgbm
pkill -9 -f "joblib"
pkill -9 -f jupyter
pkill -9 -f ipykernel_launcher
pkill -9 -f "python.*kernel"