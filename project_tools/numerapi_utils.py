import numerapi
from numerapi import utils
from project_tools import project_config, project_utils
from typing import List, Dict
import pandas as pd
import numpy as np

napi = numerapi.NumerAPI()


# def get_round


# depreciated
# def get_model_history(model):
#     res = napi.daily_user_performances(model)
#     res = pd.DataFrame.from_dict(res)
#     res['payoutPending'] = res['payoutPending'].astype(np.float64)
#     res['payoutSettled'] = res['payoutSettled'].astype(np.float64)
#     res['stakeValue'] = res['stakeValue'].astype(np.float64)
#     res['deltaRatio'] = res['payoutPending'] / res['stakeValue']
#     res['realised_pl'] = project_utils.series_reverse_cumsum(res['payoutSettled'])
#     res['floating_pl'] = project_utils.series_reverse_cumsum(res['payoutPending']) - res['realised_pl']
#     res['current_stake'] = res['stakeValue'] - res['floating_pl']
#     rename_dict = {'stakeValue':'floating_stake'}
#     res = res.rename(columns=rename_dict)
#     # res['equity'] = res['stakeValue'] + res['floating_pl']
#     # cols = res.columns.tolist()
#     # res = res[['model'] + cols]
#
#     res['model'] = model
#     cols = ['model', 'date', 'current_stake', 'floating_stake', 'payoutPending', 'floating_pl', 'realised_pl']
#     res = res[cols]
#     return res


def get_portfolio_overview(models, onlylatest=True):
    res_df = []
    for m in models:
        # try:
        print(f'extracting information for model {m}')
        if onlylatest:
            mdf = get_model_history_v3(m).loc[0:0]
        else:
            mdf = get_model_history_v3(m)
        res_df.append(mdf)
        # except:
        #     print(f'no information for model {m} is available')
    if len(res_df)>0:
        res_df = pd.concat(res_df, axis=0)
        # res_df['date'] = res_df['date'].dt.date
        if onlylatest:
            return res_df.sort_values(by='floating_pl', ascending=False).reset_index(drop=True)
        else:
            return res_df.reset_index(drop=True)
    else:
        return None






def get_competitions(tournament=8):
    """Retrieves information about all competitions
    Args:
        tournament (int, optional): ID of the tournament, defaults to 8
            -- DEPRECATED there is only one tournament nowadays
    Returns:
        list of dicts: list of rounds
        Each round's dict contains the following items:
            * datasetId (`str`)
            * number (`int`)
            * openTime (`datetime`)
            * resolveTime (`datetime`)
            * participants (`int`): number of participants
            * prizePoolNmr (`decimal.Decimal`)
            * prizePoolUsd (`decimal.Decimal`)
            * resolvedGeneral (`bool`)
            * resolvedStaking (`bool`)
            * ruleset (`string`)
    Example:
        >>> NumerAPI().get_competitions()
        [
         {'datasetId': '59a70840ca11173c8b2906ac',
          'number': 71,
          'openTime': datetime.datetime(2017, 8, 31, 0, 0),
          'resolveTime': datetime.datetime(2017, 9, 27, 21, 0),
          'participants': 1287,
          'prizePoolNmr': Decimal('0.00'),
          'prizePoolUsd': Decimal('6000.00'),
          'resolvedGeneral': True,
          'resolvedStaking': True,
          'ruleset': 'p_auction'
         },
          ..
        ]
    """
    # self.logger.info("getting rounds...")

    query = '''
        query($tournament: Int!) {
          rounds(tournament: $tournament) {
            number
            resolveTime
            openTime
            resolvedGeneral
            resolvedStaking
          }
        }
    '''
    arguments = {'tournament': tournament}
    result = napi.raw_query(query, arguments)
    rounds = result['data']['rounds']
    # convert datetime strings to datetime.datetime objects
    for r in rounds:
        utils.replace(r, "openTime", utils.parse_datetime_string)
        utils.replace(r, "resolveTime", utils.parse_datetime_string)
        utils.replace(r, "prizePoolNmr", utils.parse_float_string)
        utils.replace(r, "prizePoolUsd", utils.parse_float_string)
    return rounds


def daily_submissions_performances(username: str) -> List[Dict]:
    """Fetch daily performance of a user's submissions.
    Args:
        username (str)
    Returns:
        list of dicts: list of daily submission performance entries
        For each entry in the list, there is a dict with the following
        content:
            * date (`datetime`)
            * correlation (`float`)
            * roundNumber (`int`)
            * mmc (`float`): metamodel contribution
            * fnc (`float`): feature neutral correlation
            * correlationWithMetamodel (`float`)
    Example:
        >>> api = NumerAPI()
        >>> api.daily_user_performances("uuazed")
        [{'roundNumber': 181,
          'correlation': -0.011765912,
          'date': datetime.datetime(2019, 10, 16, 0, 0),
          'mmc': 0.3,
          'fnc': 0.1,
          'correlationWithMetamodel': 0.87},
          ...
        ]
    """
    query = """
              query($username: String!) {
                v2UserProfile(username: $username) {
                  dailySubmissionPerformances {
                    date
                    correlation
                    corrPercentile
                    roundNumber
                    mmc
                    mmcPercentile
                    fnc
                    fncPercentile
                    correlationWithMetamodel
                  }
                }
              }
            """
    arguments = {'username': username}
    data = napi.raw_query(query, arguments)['data']['v2UserProfile']
    performances = data['dailySubmissionPerformances']
    # convert strings to python objects
    for perf in performances:
        utils.replace(perf, "date", utils.parse_datetime_string)
    # remove useless items
    performances = [p for p in performances
                    if any([p['correlation'], p['fnc'], p['mmc']])]
    return performances


def daily_submissions_performances_V3(modelname: str) -> List[Dict]:
    query = """
              query($modelName: String!) {
                v3UserProfile(modelName: $modelName) {
                    roundModelPerformances{
                        roundNumber
                        roundResolveTime
                        corr
                        corrPercentile
                        mmc
                        mmcMultiplier
                        mmcPercentile
                        tc
                        tcPercentile
                        tcMultiplier
                        fncV3
                        fncV3Percentile
                        corrWMetamodel
                        payout
                        roundResolved
                        roundResolveTime
                        corrMultiplier
                        mmcMultiplier
                        selectedStakeValue
                    }
                    stakeValue
                    nmrStaked
                }
              }
            """
    arguments = {'modelName': modelname}
    data = napi.raw_query(query, arguments)['data']['v3UserProfile']
    performances = data['roundModelPerformances']
    # convert strings to python objects
    for perf in performances:
        utils.replace(perf, "date", utils.parse_datetime_string)
    # remove useless items
    performances = [p for p in performances
                    if any([p['corr'], p['tc'], p['mmc']])]
    return performances


def get_lb_models(limit=20000, offset=0):
    query = """
           query($limit: Int, $offset: Int){
               v2Leaderboard(limit:$limit, offset:$offset){
                   username
               }           
           }
           """
    arguments = {'limit':limit, 'offset':offset}
    data = napi.raw_query(query, arguments)['data']['v2Leaderboard']
    model_list = [i['username'] for i in data]
    return model_list



def get_round_model_performance(roundNumber: int, model: str):
    query = """
              query($roundNumber: Int!, $username: String!) {
                  roundSubmissionPerformance(roundNumber: $roundNumber, username: $username) {
                      corrMultiplier
                      mmcMultiplier                      
                      roundDailyPerformances{
                          correlation
                          mmc
                          corrPercentile
                          mmcPercentile
                          payoutPending
                       }
                       selectedStakeValue
                   }
              }
            """
    arguments = {'roundNumber': roundNumber,'username': model}
    data = napi.raw_query(query, arguments)['data']['roundSubmissionPerformance']
    
    if pd.isnull(data['roundDailyPerformances'][0]['payoutPending']):
        latest_performance = data['roundDailyPerformances'][1] #[-1] ### issue with order
    else:
        latest_performance = data['roundDailyPerformances'][0] #[-1] ### issue with order
    res = {}
    res['model'] = model
    res['roundNumber'] = roundNumber
    res['corrMultiplier'] = data['corrMultiplier']
    res['mmcMultiplier'] = data['mmcMultiplier']
    res['selectedStakeValue'] = data['selectedStakeValue']
    for key in latest_performance.keys():
        res[key] = latest_performance[key]
    return res




def get_user_profile(username: str) -> List[Dict]:
    """Fetch daily performance of a user's submissions.
    Args:
        username (str)
    Returns:
        list of dicts: list of daily submission performance entries
        For each entry in the list, there is a dict with the following
        content:
            * date (`datetime`)
            * correlation (`float`)
            * roundNumber (`int`)
            * mmc (`float`): metamodel contribution
            * fnc (`float`): feature neutral correlation
            * correlationWithMetamodel (`float`)
    Example:
        >>> api = NumerAPI()
        >>> api.daily_user_performances("uuazed")
        [{'roundNumber': 181,
          'correlation': -0.011765912,
          'date': datetime.datetime(2019, 10, 16, 0, 0),
          'mmc': 0.3,
          'fnc': 0.1,
          'correlationWithMetamodel': 0.87},
          ...
        ]
    """
    query = """
              query($username: String!) {
                v2UserProfile(username: $username) {
                  dailySubmissionPerformances {
                    date
                    correlation
                    corrPercentile
                    roundNumber
                    mmc
                    mmcPercentile
                    fnc
                    fncPercentile
                    correlationWithMetamodel
                  }
                }
              }
            """
    arguments = {'username': username}
    data = napi.raw_query(query, arguments)['data']#['v2UserProfile']
    # performances = data['dailySubmissionPerformances']
    # # convert strings to python objects
    # for perf in performances:
    #     utils.replace(perf, "date", utils.parse_datetime_string)
    # # remove useless items
    # performances = [p for p in performances
    #                 if any([p['correlation'], p['fnc'], p['mmc']])]
    return data


def download_dataset(filename: str, dest_path: str = None,
                     round_num: int = None) -> None:
    """ Download specified file for the current active round.

    Args:
        filename (str): file to be downloaded
        dest_path (str, optional): complate path where the file should be
            stored, defaults to the same name as the source file
        round_num (int, optional): tournament round you are interested in.
            defaults to the current round
        tournament (int, optional): ID of the tournament, defaults to 8

    Example:
        >>> filenames = NumerAPI().list_datasets()
        >>> NumerAPI().download_dataset(filenames[0]}")
    """
    if dest_path is None:
        dest_path = filename

    query = """
    query ($filename: String!
           $round: Int) {
        dataset(filename: $filename
                round: $round)
    }
    """
    args = {'filename': filename, "round": round_num}

    dataset_url = napi.raw_query(query, args)['data']['dataset']
    utils.download_file(dataset_url, dest_path, show_progress_bars=True)
    
    
    
# function using V3UserProfile

def model_payout_history(model):
    napi = numerapi.NumerAPI()
    query = """
              query($model: String!) {
                  v3UserProfile(modelName: $model) {
                        roundModelPerformances{
                            payout
                            roundNumber
                            roundResolved
                            roundResolveTime
                            corrMultiplier
                            mmcMultiplier
                            selectedStakeValue
                        }
                        stakeValue
                        nmrStaked
                   }
              }
            """
    arguments = {'model': model}
    payout_info = napi.raw_query(query, arguments)['data']['v3UserProfile']['roundModelPerformances']
    payout_info = pd.DataFrame.from_dict(payout_info)
    payout_info = payout_info[~pd.isnull(payout_info['payout'])].reset_index(drop=True)
    return payout_info


def get_model_history_v3(model):
    res = model_payout_history(model)
    res = pd.DataFrame.from_dict(res)
    res['payout'] = res['payout'].astype(np.float64)
    res['current_stake'] = res['selectedStakeValue'].astype(np.float64)
    res['payout_cumsum'] = project_utils.series_reverse_cumsum(res['payout'])
    res['date'] = pd.to_datetime(res['roundResolveTime']).dt.date

    res['realised_pl'] = res['payout_cumsum']
    latest_realised_pl = res[res['roundResolved'] == True]['payout_cumsum'].values[0]
    res.loc[res['roundResolved'] == False, 'realised_pl'] = latest_realised_pl

    res['floating_pl'] = 0
    payoutPending_values = res[res['roundResolved'] == False]['payout'].values
    payoutPending_cumsum = payoutPending_values[::-1].cumsum()[::-1]
    res.loc[res['roundResolved'] == False, 'floating_pl'] = payoutPending_cumsum

    res['model'] = model
    #     res['floating_pl'] = res['current_stake'] + res['payoutPending']
    res['floating_stake'] = res['current_stake'] + res['floating_pl']
    cols = ['model', 'date', 'current_stake', 'floating_stake', 'payout', 'floating_pl', 'realised_pl', 'roundResolved',
            'roundNumber']
    res = res[cols]
    return res






