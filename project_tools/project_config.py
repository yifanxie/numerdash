import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

DATETIME_FORMAT1 = '%Y%m%d%H%M'
DATETIME_FORMAT2 = '%Y/%m/%d %H:%M'
DATETIME_FORMAT3 = '%Y-%m-%d'
SAVE_LOCAL_COPY = True

BENCHMARK_MODELS = ['integration_test', 'integration_test_7'] #'budbot_7'] #'integration_test_7'
FEATURE_PATH = './feature_data/'
MODEL_ROUND_RESULT_FILE = './feature_data/model_round_result.pkl'
MODEL_DAILY_RESULT_FILE = './feature_data/model_daily_result.pkl'
NUMERATI_FILE = './feature_data/numerati_data.pkl'

NUMERATI_URL = 'https://raw.githubusercontent.com/woobe/numerati/master/data.csv'





