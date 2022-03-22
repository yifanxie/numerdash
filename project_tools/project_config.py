import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

DATETIME_FORMAT1 = '%Y%m%d%H%M'
DATETIME_FORMAT2 = '%Y/%m/%d %H:%M'
DATETIME_FORMAT3 = '%Y-%m-%d'
SAVE_LOCAL_COPY = True

BENCHMARK_MODELS = ['integration_test', 'integration_test_7'] #'budbot_7'] #'integration_test_7'
MODEL_ROUND_RESULT_FILE = '../feature_data/model_round_result.pkl'
MODEL_DAILY_RESULT_FILE = '../feature_data/model_daily_result.pkl'

NUMERATI_URL = 'https://raw.githubusercontent.com/woobe/numerati/master/data.csv'
NUMERATI_FILE = '../feature_data/numerati_data.pkl'
FEATURE_PATH = '../feature_data/'



# to be discarded
MODEL_NAMES = ['yxbot', 'yxbot2', 'sforest_baihu', 'stree_qinlong', 'flyingbus_mcv6', 'starry_night','fish_and_chips', 'rogue_planet', 'three_body_problem', 'grinning_cat', 'schrodingers_cat', 'omega_weapon', 'ifirit','dark_bahamut', 'wen_score',  'qinlong', 'baihu','marlboro', 'hell_cerberus', 'fuxi', 'roci_fuxi', 'kupo_mcv7', 'yxbot_mcv2', 'yxbot_mcv10']


NEW_MODEL_NAMES = ['yxbot3_m15', 'yxbot4_m23', 'yxbot5', 'yxbot6_m16', 'yxbot7_m17', 'yxbot_a10b8', 'yxbot9_m24', 'yxbot_a10', 'yxbot_a10xu', 'yxbot_a10bk','yxbot_a11', 'yxbot_a12', 'yxbot_ultima_weapon', 'yxbot_valkyrie', 'yxbot_bearmate', 'yxbot_dracula','yxbot_a13', 'yxbot_a14', 'yxbot15_zhuque', 'yxbot_redhare', 'yxbot_a15', 'yxbot18_m25', 'yxbot11_x302']

# flyingbus

TOP_LB = ['mdl3', 'nescience', 'sapphirescipionyx','quantaquetzalcoatlus', 'anna13', 'mercuryai', 'uuazed6', 'rosetta', 'sinookas']


TP3M = ['ageonsen', 'davebaty', 'wallingford_nut', 'filipstefano2', 'davat6', 'lions', 'wsw', 'lottery_of_babylon', 'kup_choy_n', 'pinky_and_the_brain']


TP1Y = ['hiryuu', 'victoria', 'benben11', 'usigma7', 'crystal_sphere', 'era__mix__2000', 'rgb_alpha', 'smokh', 'shoukaku', 'stables', 'deepnum', 'botarai', 'zuikaku', 'kond']


ARBITRAGE_MODELS = ['arbitrage', 'arbitrage2', 'arbitrage3', 'arbitrage4', 'leverage', 'leverage2', 'leverage3', 'culebracapital', 'culebracapital2', 'culebracapital3']


IAAI_MODELS = ['ia_ai',  'the_aijoe4','i_like_the_coin_08', 'i_like_the_coin_09', 'i_like_the_coin_10']


RESTRADE_MODELS = ['restrading', 'restrading2', 'restrading3', 'restrading4', 'restrading5', 'restrading6', 'restrading7', 'restrading8', 'restrading9']

MCV_MODELS = ['mcv', 'mcv2', 'mcv3', 'mcv4', 'mcv5','mcv6','mcv7','mcv8','mcv9','mcv10','mcv11','mcv12','mcv13']
MCV_NEW_MODELS = ['mcv14', 'mcv15', 'mcv16', 'mcv17', 'mcv18', 'mcv19', 'mcv20', 'mcv21', 'mcv22', 'mcv23', 'mcv24', 'mcv25', 'mcv26', 'mcv27', 'mcv28', 'mcv29', 'mcv30', 'mcv31', 'mcv32', 'mcv33', 'mcv34', 'mcv35', 'mcv36', 'mcv37', 'mcv38', 'mcv39', 'mcv40', 'mcv41', 'mcv42', 'mcv43', 'mcv44', 'mcv45', 'mcv46', 'mcv47', 'mcv48', 'mcv49', 'mcv50']

