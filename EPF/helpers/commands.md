

# pull (delete local files not in S3)
aws s3 sync s3://forecasting-nem-dd . --region ap-southeast-2 
--exclude ".git/*" 
--exclude "2_Features_build/Feature_data/*" 
--delete


# push (delete s3 files not in local)
aws s3 sync . s3://forecasting-nem-dd --region ap-southeast-2 
--exclude 
".git/*" 
--exclude "2_Features_build/Feature_data/*" 
--delete

# pull just one s3 folder
aws s3 sync s3://forecasting-nem-dd/1_Dataset/Processed_data ./1_Dataset/Processed_data --region ap-southeast-2

# kill all python kernels
pkill -9 -f ipykernel
pkill -9 -f lightgbm
pkill -9 -f "joblib"
pkill -9 -f jupyter
pkill -9 -f ipykernel_launcher
pkill -9 -f "python.*kernel"


# Create venvs

## Windows (PowerShell)

### Main environment
python3.14 -m venv C:\Users\danie\.venvs\venv-main

C:\Users\danie\.venvs\venv-main\Scripts\Activate.ps1

pip install -r requirements-main.txt


### Subprocess environment
python3.11 -m venv C:\Users\danie\.venvs\venv-subprocess

C:\Users\danie\.venvs\venv-subprocess\Scripts\Activate.ps1

pip install -r requirements-subprocess.txt



## Linux

### Main environment
python3.14 -m venv ~/.venvs/venv-main

source ~/.venvs/venv-main/bin/activate

pip install -r requirements-main.txt


### Subprocess environment
python3.11 -m venv ~/.venvs/venv-subprocess

source ~/.venvs/venv-subprocess/bin/activate

pip install -r requirements-subprocess.txt