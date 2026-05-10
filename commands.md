# push
aws s3 sync . s3://forecasting-nem-dd --region ap-southeast-2 --exclude ".venv*" --exclude ".git/*"

# pull
aws s3 sync s3://forecasting-nem-dd . --region ap-southeast-2 --exclude ".venv*"