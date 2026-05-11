# push
aws s3 sync . s3://forecasting-nem-dd --region ap-southeast-2 --exclude "2_Features build/Feature_data/*"

# pull
aws s3 sync s3://forecasting-nem-dd . --region ap-southeast-2 --exclude "2_Features build/Feature_data/*"




