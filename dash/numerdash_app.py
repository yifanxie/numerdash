import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from project_tools import project_utils, project_config, numerapi_utils
import warnings
import plotly.express as px
import json
warnings.filterwarnings("ignore")
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit import caching
import time
import traceback
import datetime

st.set_page_config(layout='wide')
get_benchmark_data = True

# get_dailyscore = True




def sidebar_data_picker():
    st.sidebar.subheader('Model Data Picker')
    top_lb = st.sidebar.checkbox('top LB by corr', value=True)
    top_tp3m = st.sidebar.checkbox('most profitable 3 month', value=True)
    top_tp1y = st.sidebar.checkbox('most profitable 1 year', value=True)
    special_list = st.sidebar.checkbox('model from specific users', value=True)
    return top_lb, top_tp3m, top_tp1y, special_list


# to be removed
def model_data_picker_bak(values = None):
    if values is None:
        values = [True, True, True, True, True, True]
    model_dict = {}
    st.sidebar.subheader('Model Data Picker')
    # top_lb = st.sidebar.checkbox('top LB by corr', value=values[0])
    # top_tp3m = st.sidebar.checkbox('most profitable 3 month', value=values[1])
    top_tp1y = st.sidebar.checkbox('most profitable 1 year', value=values[2])
    special_list = st.sidebar.checkbox('model from specific users', value=values[3])
    benchmark_list = st.sidebar.checkbox('benchmark models', value=values[4])
    default_list = st.sidebar.checkbox('default models', value=values[5])
    # if top_lb:
    #     model_dict['top_corr'] = project_config.TOP_LB
    # if top_tp3m:
        # model_dict['top_3m'] = project_config.TP3M
    if top_tp1y:
        model_dict['top_1y'] = project_config.TP1Y
    if benchmark_list:
        model_dict['benchmark'] = project_config.BENCHMARK_MODELS
    if special_list:
        model_dict['iaai'] = project_config.IAAI_MODELS
        # model_dict['arbitrage'] = project_config.ARBITRAGE_MODELS
        # model_dict['mm'] = project_config.MM_MODELS
        # model_dict['restrade'] = project_config.RESTRADE_MODELS

    if default_list:
        model_dict['yx'] = project_config.MODEL_NAMES + project_config.NEW_MODEL_NAMES
        model_dict['mcv'] = project_config.MCV_MODELS + project_config.MCV_NEW_MODELS
    return model_dict


# to be removed
def model_fast_picker_bak(models):
    text_content = '''
                    fast model picker by CSV string.
                    example: "model1, model2, model3" 
                   '''
    text = st.sidebar.text_area(text_content)
    result_models = []
    if len(text)>0:
        csv_parts = text.split(',')
        for s in csv_parts:
            m = s.strip()
            if m in models:
                result_models.append(m)
    return list(dict.fromkeys(result_models))



def default_model_picker():
    picked_models = {}
    if os.path.isfile('default_models.json'):
        default_models_dict = project_utils.load_json('default_models.json')
        for key in default_models_dict.keys():
            picked_models[key] = default_models_dict[key]
    if os.path.isfile('user_models.json'):
        user_models_dict = project_utils.load_json('user_models.json')
        for key in user_models_dict.keys():
            picked_models[key] = user_models_dict[key]
    return picked_models


def model_fast_picker(models):
    text_content = '''
                    fast model picker by CSV string.
                    example: "model1, model2, model3" 
                   '''
    text = st.sidebar.text_area(text_content)
    result_models = []
    if len(text)>0:
        csv_parts = text.split(',')
        for s in csv_parts:
            m = s.strip()
            if m not in models:
                result_models.append(m)
    return list(dict.fromkeys(result_models))






def generate_round_table(data, row_cts, c, r, sortcol='corrmmc'):
    # rounds = data
    # row_cts[c].write(2*r+c)
    latest_round = int(data['roundNumber'].max())
    earliest_round =  int(data['roundNumber'].min())
    suggest_round = int(latest_round - (2*r+c))
    select_round = row_cts[c].slider('select a round', earliest_round, latest_round, suggest_round, 1)
    # row_cts[c].write(select_round)
    round_data = data[data['roundNumber']==select_round].sort_values(by=sortcol, ascending=False).reset_index(drop=True)
    # round_data = round_data[round_data['model'].isin(models)].reset_index(drop=True)
    latest_date = round_data['date'].values[0]
    row_cts[c].write(f'round: {select_round}, date: {latest_date}')
    row_cts[c].dataframe(round_data.drop(['roundNumber', 'date'], axis=1), height=max_table_height-100)






def generate_dailyscore_metrics(data, row_cts, c, r):
    # row_cts[c].write([r, c, 2*r+c])
    select_metric = row_cts[c].selectbox("", list(id_metric_opt.keys()), index=2*r+c, format_func=lambda x: id_metric_opt[x])
    latest_round = int(data['roundNumber'].max())
    earliest_round = int(data['roundNumber'].min())
    score = id_metric_score_dic[select_metric]
    df = project_utils.calculate_rounddailysharpe_dashboard(data, latest_round, earliest_round, score).sort_values(by='sos', ascending=False)
    row_cts[c].dataframe(df, height=max_table_height-100)
    pass

def get_roundmetric_data(data):
    numfeats1 = ['corr', 'mmc', 'corrmmc', 'corr2mmc']
    stat1 = ['sum', 'mean', 'count',
             {'sharpe': project_utils.get_array_sharpe}]  # {'ptp':np.ptp}]#{'sharp':project_utils.get_array_sharpe}]
    numfeats2 = ['corr_pct', 'mmc_pct', 'cmavg_pct', 'c2mavg_pct']
    stat2 = ['mean']#, {'sharp': project_utils.get_array_sharpe}]

    roundmetric_agg_rcp = [
        [['model'], numfeats1, stat1],
        [['model'], numfeats2, stat2]
    ]

    res = project_utils.groupby_agg_execution(roundmetric_agg_rcp, data)['model']
    rename_dict = {}
    for c in res.columns.tolist():
        if c != 'model':
            rename_dict[c] = c[6:] # remove 'model_' in column name
    res.rename(columns = rename_dict, inplace=True)
    return res


def generate_round_metrics(data, row_cts, c, r):
    select_metric = row_cts[c].selectbox("", list(roundmetric_opt.keys()), index=2*r+c, format_func=lambda x: roundmetric_opt[x])
    cols = ['model']
    # st.write(select_metric)
    # st.write(data.columns.tolist())
    for col in data.columns.tolist():
        if select_metric =='corrmmc':
            if (f'{select_metric}_' in col) or ('cmavg_' in col):
                cols += [col]
        elif select_metric =='corr2mmc':
            if (f'{select_metric}_' in col) or ('c2mavg_' in col):
                cols += [col]
        else:
            if (f'{select_metric}_' in col) and (not('corrmmc' in col)) and (not('corr2mmc' in col)):
                cols+= [col]

    if select_metric != 'pct':
        sort_col = select_metric+'_sharpe'
    else:
        sort_col = 'cmavg_pct_mean'
    view_data = data[cols].sort_values(by=sort_col, ascending=False)
    row_cts[c].dataframe(view_data)
    pass


def dailyscore_chart(data, row_cts, c, r, select_metric):
    latest_round = int(data['roundNumber'].max())
    earliest_round =  int(data['roundNumber'].min())
    suggest_round = int(latest_round - (2*r+c))
    select_round = row_cts[c].slider('select a round', earliest_round, latest_round, suggest_round, 1)
    data = data[data['roundNumber']==select_round]
    if len(data)>0:
        fig = chart_pxline(data, 'date', y=select_metric, color='model', hover_data=list(histtrend_opt.keys()))
        row_cts[c].plotly_chart(fig, use_container_width=True)
    else:
        row_cts[c].info('no data was found for the selected round')
    pass


def generate_live_round_stake(data, row_cts, c, r):
    latest_round = int(data['roundNumber'].max())
    select_round = int(latest_round - (2*r+c))
    select_data = data[data['roundNumber']==select_round].reset_index(drop=True)
    if len(select_data)>0:
        payout_sum = select_data['payout'].sum().round(3)
        stake_sum = select_data['stake'].sum().round(3)
        if payout_sum >= 0:
            payout_color = 'green'
        else:
            payout_color = 'red'

        space = '&nbsp;'*5
        content_str = f'#### Round: {select_round}{space}Stake: {stake_sum}{space}Payout: <span style="color:{payout_color}">{payout_sum}</span> NMR'
        row_cts[c].markdown(content_str, unsafe_allow_html=True)
        select_data = select_data.drop(['roundNumber'], axis=1).sort_values(by='payout', ascending=False)
        row_cts[c].dataframe(select_data, height=max_table_height-100)



def round_view(data, select_perview, select_metric=None):
    num_cols = 2
    num_rows = 2
    for r in range(num_rows):
        row_cts = st.columns(num_cols)
        for c in range(num_cols):
            if select_perview=='round_result':
                generate_round_table(data, row_cts, c, r)
            if select_perview=='dailyscore_metric':
                generate_dailyscore_metrics(data, row_cts, c, r)
            if select_perview=='round_metric':
                generate_round_metrics(data, row_cts, c, r)
            if select_perview=='dailyscore_chart':
                dailyscore_chart(data, row_cts, c, r, select_metric)
            if select_perview=='live_round_stake':
                 generate_live_round_stake(data, row_cts, c, r)


def score_overview():
    models = []
    data = []
    benchmark_opt = st.sidebar.checkbox('download default models', value=True)

    model_selection = st.empty()
    if benchmark_opt:
        model_dict = default_model_picker()
        for k in model_dict.keys():
            models += model_dict[k]
    models = models + model_fast_picker(models)
    # if len(models)>0:
        # model_selection = st.sidebar.multiselect('select models', models, default=models)
    st.sidebar.subheader('Choose a Table View')
    select_perview = st.sidebar.selectbox("", list(tbl_opt.keys()), index=0, format_func=lambda x: tbl_opt[x])
    if len(models)>0:
        model_selection.multiselect('selected models', models, default=models)


def data_operation():
    # top_lb, top_tp3m, top_tp1y, special_list = sidebar_data_picker()
    latest_round = project_utils.latest_round
    models = []
    benchmark_opt = st.sidebar.checkbox('download default models', value=True)
    if benchmark_opt:
        model_dict = default_model_picker()
        for k in model_dict.keys():
            models += model_dict[k]
    models = models + model_fast_picker(models)
    if len(models)>0:
        model_selection = st.multiselect('select models', models, default=models)
    suggest_min_round = 182 #latest_round-50
    min_round, max_round = st.slider('select tournament rounds', 200, latest_round, (suggest_min_round, latest_round), 1)
    roundlist = [i for i in range(max_round, min_round-1, -1)]
    download = st.button('download data of selected models')
    st.sidebar.subheader('configuration')
    show_info=st.sidebar.checkbox('show background data', value=False)
    # update_numeraiti_data = st.sidebar.checkbox('update numerati data', value=True)
    # update_model_data = st.sidebar.checkbox('update model data', value=True)
    # update_model_data =

    model_df = get_saved_data()
    if download and len(model_selection)>0:
        # if update_model_data:
        with st.spinner('downloading model round results'):
            model_df = []
            model_df = download_model_round_result(model_selection, roundlist, show_info)

    prjreload = st.sidebar.button('reload config')
    if prjreload:
        project_utils.reload_project()
    if len(model_df)>0:
        rename_dict = {'corrPercentile': 'corr_pct', 'correlation':'corr', 'corrWMetamodel':'corr_meta', 'mmcPercentile':'mmc_pct', 'tcPercentile':'tc_pct'}
        model_df.rename(columns=rename_dict, inplace=True)
        model_df['corrmmc'] = model_df['corr'] + model_df['mmc']
        model_df['corr2mmc'] = model_df['corr'] + 2*model_df['mmc']
        model_df['cmavg_pct'] = (model_df['corr_pct'] + model_df['mmc_pct'])/2
        model_df['c2mavg_pct'] = (model_df['corr_pct'] + 2*model_df['mmc_pct'])/3
        ord_cols = ['model','corr', 'corr_pct', 'mmc', 'mmc_pct', 'corrmmc', 'cmavg_pct', 'corr_meta', 'tc', 'tc_pct', 'corr2mmc','c2mavg_pct', 'roundNumber']
        model_df = model_df[ord_cols]
        if project_config.SAVE_LOCAL_COPY:
            project_utils.pickle_data(project_config.MODEL_ROUND_RESULT_FILE, model_df)
        st.session_state['model_data'] = model_df

    if show_info:
        st.text('list of models being tracked')
        st.write(model_dict)
        try:
            st.write(st.session_state['model_data'].shape)
            st.write(model_df.head(5))
        except:
            st.write('model data was not retrieved')

    if len(model_df)>0:
        get_performance_data_status(model_df)
    return None

def get_saved_data():
    res = []
    if os.path.isfile(project_config.MODEL_ROUND_RESULT_FILE):
        res = project_utils.load_data(project_config.MODEL_ROUND_RESULT_FILE)
        st.session_state['model_data'] = res
    return res

def get_performance_data_status(df):
    st.sidebar.subheader('model data summary')
    # latest_date = df['date'][0].strftime(project_config.DATETIME_FORMAT3)
    model_num = df['model'].nunique()
    round_num = df['roundNumber'].nunique()
    latest_round = df['roundNumber'].max()
    # st.sidebar.text(f'latest date: {latest_date}')
    st.sidebar.text(f'number of models: {model_num}')
    st.sidebar.text(f'number of rounds: {round_num}')
    st.sidebar.text(f'latest round: {latest_round}')
    return None


def download_model_round_result(models, roundlist, show_info):
    model_df = []
    model_dfs = []
    my_bar = st.progress(0.0)
    my_bar.progress(0.0)
    percent_complete = 0.0
    for i in range(len(models)):
        message = ''
        try:
            model_res = numerapi_utils.daily_submissions_performances_V3(models[i])
            if len(model_res) > 0:
                cols = ['model'] + list(model_res[0].keys())
                model_df = pd.DataFrame(model_res)
                model_df['model'] = models[i]
                model_df = model_df[cols]
                model_dfs.append(model_df)
            else:
                message = f'no result found for model {models[i]}'
        except Exception:
            # if show_info:
            #     st.write(f'error while getting result for {models[i]}')
            except_msg = traceback.format_exc()
            message = f'error while getting result for {models[i]}: {except_msg}'
        if show_info and len(message) > 0:
            st.info(message)
        percent_complete += 1 / len(models)
        if i == len(models) - 1:
            percent_complete = 1.0
        time.sleep(0.1)
        my_bar.progress(percent_complete)
        model_df = pd.concat(model_dfs, axis=0).sort_values(by=['roundNumber'], ascending=False).reset_index(drop=True)
        model_df = model_df[model_df['roundNumber'].isin(roundlist)].reset_index(drop=True)
    return model_df

def chart_pxline(data, x, y, color, hover_data=None, x_range=None):
    fig = px.line(data, x=x, y=y, color=color, hover_data=hover_data)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', height = max_height, margin=dict(l=0, r=10, t=20, b=20))
    fig.update_xaxes(showgrid=False, range=x_range)
    fig.update_yaxes(gridcolor='grey')
    return fig


def roundresult_chart(data, model_selection):

    round_data = data[data['model'].isin(model_selection)].drop_duplicates(['model', 'roundNumber'],                                                                           keep='first').reset_index(drop=True)
    min_round = int(round_data['roundNumber'].min())
    max_round = int(round_data['roundNumber'].max())
    suggest_min_round = max_round - 20
    if min_round == max_round:
        min_round = max_round - 20

    min_selectround, max_selectround = st.slider('select plotting round range', min_round, max_round,
                                                 (suggest_min_round, max_round), 1)

    # min_selectround, max_selectround = placeholder.slider('select plotting round range', min_round, max_round,
    #                                              (suggest_min_round, max_round), 1)

    select_metric = st.selectbox('Choose a metric', list(histtrend_opt.keys()), index=0,
                                 format_func=lambda x: histtrend_opt[x])
    round_range = [min_selectround, max_selectround]
    round_list = [r for r in range(min_selectround, max_selectround + 1)]
    round_data = round_data[round_data['roundNumber'].isin(round_list)]
    mean_df = round_data.groupby(['model'])[select_metric].agg('mean').reset_index()
    mean_df[f'model avg.'] = mean_df['model'] + ': ' + mean_df[select_metric].round(5).astype(str)
    mean_df['mean'] = mean_df[select_metric]
    merge_cols = ['model', 'model avg.', 'mean']
    round_data = round_data.merge(right=mean_df[merge_cols], on='model', how='left').sort_values(by=['mean','model', 'roundNumber'], ascending=False)
    fig = chart_pxline(round_data, 'roundNumber', y=select_metric, color='model avg.', hover_data=list(histtrend_opt.keys())+['date'],x_range=round_range)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        dailyscore_data = data[(data['model'].isin(model_selection)) & data['roundNumber'].isin(round_list)].reset_index(drop=True)
        dailyscore_data = dailyscore_data.merge(right=mean_df[merge_cols], on='model', how='left').sort_values(
            by=['mean', 'model', 'roundNumber'], ascending=False)
        round_view(dailyscore_data, 'dailyscore_chart', select_metric)

    else:
        st.text(f'No data available for models: {models}')




def histtrend():
    # default_models = ['yxbot']
    # models = default_models.copy()
    models = []
    model_selection = []
    model_dict = model_data_picker(values=[False, False, False, False, True, True])
    for k in model_dict.keys():
        if model_dict[k] not in models:
            models += model_dict[k]

    default_models = model_fast_picker(models)
    if len(models)>0:
        if len(default_models)==0:
            default_models = [models[0]]
        model_selection = st.sidebar.multiselect('select models for chart', models, default=default_models)


    if os.path.isfile(project_config.DASHBOARD_MODEL_RESULT_FILE) and len(model_selection)>0:
        data = project_utils.load_data(project_config.DASHBOARD_MODEL_RESULT_FILE)
        roundresult_chart(data, model_selection)

        # fig = px.line(df, x='roundNumber', y='corr', color='model', hover_data=['corr_pct'])
        # st.write(model_selection)
    else:
        if len(model_selection)==0:
            st.info('please select some models from the dropdown list')
        else:
            st.info('model result data file missing, or no model is selected')

    # st.write(models)



def model_evaluation():
    models = []
    model_selection = []
    model_dict = model_data_picker_bak(values=[True, True, True, True, True, True])
    mean_scale = [-0.05, 0.1]
    count_scale = [1, 50]
    sharpe_scale = [-0.2, 3]
    pct_scale = [0, 1]
    radar_scale = [0, 5]

    for k in model_dict.keys():
        if model_dict[k] not in models:
            models += model_dict[k]

    default_models = model_fast_picker(models)
    if len(models)>0:
        if len(default_models)==0:
            if 'integration_test' in models:
                default_models = 'integration_test'
            else:
                default_models = [models[0]]

        model_selection = st.sidebar.multiselect('select models for chart', models, default=default_models)

    if os.path.isfile(project_config.DASHBOARD_MODEL_RESULT_FILE) and len(model_selection)>0:
        data = project_utils.load_data(project_config.DASHBOARD_MODEL_RESULT_FILE)
        round_data = data[data['model'].isin(model_selection)].drop_duplicates(['model', 'roundNumber'],keep='first').reset_index(drop=True)
        min_round = int(round_data['roundNumber'].min())
        max_round = int(round_data['roundNumber'].max())
        suggest_min_round = max_round - 20
        if min_round == max_round:
            min_round = max_round - 20

        min_selectround, max_selectround = st.slider('select plotting round range', min_round, max_round,
                                                     (suggest_min_round, max_round), 1)
        round_list = [r for r in range(min_selectround, max_selectround+1)]
        defaultlist = ['corr_sharpe', 'mmc_sharpe',  'corr2mmc_sharpe','corr_mean', 'mmc_mean', 'corr2mmc_mean', 'count']

        select_metrics = st.multiselect('Metric Selection', list(model_eval_opt.keys()),
                                     format_func=lambda x: model_eval_opt[x], default=defaultlist)

        use_dailymetrics = ('id_corr_sharpe' in select_metrics) or (('id_mmc_sharpe' in select_metrics)) or ('id_corrmmc_sharpe' in select_metrics)
        if use_dailymetrics:
            st.write('use daily metrics')

        round_data = round_data[round_data['roundNumber'].isin(round_list)].reset_index(drop=True)
        #'need normalised radar chart + tabular view here
        roundmetric_df = get_roundmetric_data(round_data).sort_values(by='corrmmc_sharpe', ascending=False).reset_index(drop=True)

        radarmetric_df = roundmetric_df.copy(deep=True)
        for col in select_metrics:
            if 'mean' in col:
                use_scale = mean_scale
            if 'sharpe' in col:
                use_scale = sharpe_scale
            if 'pct' in col:
                use_scale = pct_scale
            if 'count' in col:
                use_scale = count_scale
            radarmetric_df[col] = radarmetric_df[col].apply(lambda x: project_utils.rescale(x, use_scale, radar_scale))
        select_metrics_name = [model_eval_opt[i] for i in select_metrics]
        radarmetric_df.rename(columns=model_eval_opt, inplace=True)
        roundmetric_df.rename(columns=model_eval_opt, inplace=True)

        fig = go.Figure()
        for i in range(len(radarmetric_df)):
            fig.add_trace(go.Scatterpolar(
                r=radarmetric_df.loc[i, select_metrics_name].values,
                theta=select_metrics_name,
                fill='toself',
                name=radarmetric_df['model'].values[i]
            ))

        fig.update_polars(
            radialaxis=dict(visible=True, autorange=False, #type='linear',
                            range=[0,5])
        )

        fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='aliceblue',
                          height=max_height+100,
                          margin=dict(l=0, r=10, t=20, b=20), showlegend=True)

        st.plotly_chart(fig, use_container_width=True)
        st.text('Calculated Metrics')
        st.dataframe(roundmetric_df[['model'] + select_metrics_name], height=max_table_height)
        st.text('Rescaled Metrics on Chart')
        st.dataframe(radarmetric_df[['model'] + select_metrics_name], height=max_table_height)

        # st.write(select_metrics)


def get_portfolio_overview(models, onlylatest=True):
    res_df = []
    my_bar = st.progress(0.0)
    my_bar.progress(0.0)
    percent_complete = 0.0
    for i in range(len(models)):
        m = models[i]
        try:
            if onlylatest:
                # mdf = numerapi_utils.get_model_history(m).loc[0:0]
                mdf = numerapi_utils.get_model_history_v3(m).loc[0:0]
            else:
                # mdf = numerapi_utils.get_model_history(m)
                mdf = numerapi_utils.get_model_history_v3(m)
            res_df.append(mdf)
        except:
            # st.info(f'no information for model {m} is available')
            pass
        percent_complete += 1 / len(models)
        if i == len(models) - 1:
            percent_complete = 1.0
        time.sleep(0.1)
        my_bar.progress(percent_complete)
    try:
        res_df = pd.concat(res_df, axis=0)
        res_df['profitability'] = res_df['realised_pl']/(res_df['current_stake']-res_df['realised_pl'])
        cols = ['model', 'date', 'current_stake', 'floating_stake', 'floating_pl', 'realised_pl', 'profitability']

        # res_df['date'] = res_df['date'].dt.date
        if onlylatest:
            res_df = res_df.sort_values(by='floating_pl', ascending=False).reset_index(drop=True)
            return res_df[cols]
        else:
            return res_df[cols]
    except:
        return []


def get_stake_type(corr, mmc):
    if mmc>0:
        res = str(int(corr)) + 'xCORR ' + str(int(mmc)) +'xMMC'
    else:
        res = '1xCORR'
    return res


@st.cache(suppress_st_warning=True)
def get_stake_by_liverounds(models):
    latest_round_id = int(project_utils.get_latest_round_id())
    roundlist = [i for i in range(latest_round_id, latest_round_id - 5, -1)]
    res = []
    my_bar = st.progress(0.0)
    my_bar.progress(0.0)
    percent_complete = 0.0
    percent_part = 0
    for r in roundlist:
        for m in models:
            percent_complete += 1 / (len(models)*len(roundlist))
            try:
                data = numerapi_utils.get_round_model_performance(r, m)
                # print(f'successfuly extract for model {m} in round {r}')
                res.append(data)
            except:
                pass
                # print(f'no result found for model {m} in round {r}')
            if percent_part == (len(models)*len(roundlist)) - 1:
                percent_complete = 1.0
            time.sleep(0.1)
            my_bar.progress(percent_complete)
            percent_part +=1
    res_df = pd.DataFrame.from_dict(res).fillna(0)
    res_df['payoutPending'] = res_df['payoutPending'].astype(np.float64)
    res_df['selectedStakeValue'] = res_df['selectedStakeValue'].astype(np.float64)
    res_df['stake_type'] = res_df.apply(lambda x: get_stake_type(x['corrMultiplier'], x['mmcMultiplier']),axis=1)
    rename_dict = {'selectedStakeValue': 'stake', 'payoutPending': 'payout', 'correlation':'corr'}
    res_df = res_df.rename(columns=rename_dict)
    col_ord = ['model', 'roundNumber', 'stake', 'payout', 'stake_type',  'corr', 'mmc']
    return res_df[col_ord]



def get_stake_graph(data):
    numfeats = ['current_stake', 'floating_stake', 'floating_pl', 'realised_pl']
    stat1 = ['sum']
    agg_rcp = [[['date'], numfeats, stat1]]

    select_opt = st.selectbox('Select Time Span', list(stakeoverview_plot_opt.keys()), index=1, format_func=lambda x: stakeoverview_plot_opt[x])

    res = project_utils.groupby_agg_execution(agg_rcp, data)['date']
    w5delta = datetime.timedelta(weeks=5)
    w13delta = datetime.timedelta(weeks=13)
    date_w5delta = res['date'].max() - w5delta
    date_w13delta = res['date'].max() - w13delta
    y1delta = datetime.timedelta(weeks=52)
    date_y1delta = res['date'].max() - y1delta

    rename_dict = {'date_current_stake_sum': 'total_stake', 'date_floating_stake_sum': 'floating_stake',
                   'date_floating_pl_sum': 'floating_pl', 'date_realised_pl_sum': 'realised_pl'}
    res = res.rename(columns=rename_dict)
    if select_opt == '1month':
        res = res[res['date']>date_w5delta]
    elif select_opt=='3month':
        res = res[res['date']>date_w13delta]
    elif select_opt=='1year':
        res = res[res['date']>date_y1delta]
    else:
        pass

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace( go.Scatter(x=res['date'], y=res['floating_stake'], name="floating_stake"), secondary_y=False,)

    fig.add_trace(go.Scatter(x=res['date'], y=res['total_stake'], name="total_stake"),secondary_y=False,)

    fig.add_trace(go.Scatter(x=res['date'], y=res['realised_pl'], name="realised_pl"),secondary_y=True,)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.update_xaxes(showgrid=False, range=None, nticks=30)
    fig.update_yaxes(gridcolor='grey', title_text="total stake/floating stake/realised PL", secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text="realised PL", zeroline=False,secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

#
# def live_round_stakeview(data):
#     models = data
#     latest_round_id = int(project_utils.get_latest_round_id())
#     roundlist = [i for i in range(latest_round_id, latest_round_id-4, -1]


def check_session_state(key):
    # st.write(data)
    if key in st.session_state:
        return st.session_state[key]
    else:
        return None


def stake_overview():
    models = []
    model_selection = []
    model_dict = model_data_picker_bak(values=[True, True, True, True, True, True])
    for k in model_dict.keys():
        if model_dict[k] not in models:
            models += model_dict[k]

    default_models = model_fast_picker(models)
    if len(models)>0:
        model_selection = st.sidebar.multiselect('select models for chart', models, default=default_models)
    redownload_data = False
    if len(model_selection) > 0:
        if 'stake_df' not in st.session_state:
            redownload_data = True
        else:
            if set(model_selection)!=st.session_state['stake_overview_models']:
                redownload_data = True
            else:
                ovdf = st.session_state['stake_df']
        if redownload_data:
            ovdf = get_portfolio_overview(model_selection, onlylatest=False)
            print(ovdf.shape)
            st.session_state['stake_df'] = ovdf
            st.session_state['stake_overview_models'] = set(ovdf['model'].unique().tolist())

        chartdf = ovdf.copy(deep=True)
        ovdf = ovdf.drop_duplicates('model', keep='first')
        ovdf = ovdf.sort_values(by='floating_pl', ascending=False).reset_index(drop=True)
        if len(ovdf)>0:
            numerai_date = str(ovdf['date'].values[0])[0:10]
            ovdf.drop(['date'], axis=1, inplace=True)
            stake_cts = st.columns(2)
            pl_cts = st.columns(2)
            date_label = st.empty()
            get_stake_graph(chartdf)
            ovdf_exp = st.expander('', expanded=True)
            with ovdf_exp:
                st.dataframe(ovdf, height=max_table_height)
            total_current_stake = round(ovdf['current_stake'].sum(), 3)
            total_floating_stake = round(ovdf['floating_stake'].sum(), 3)
            rpl = round(ovdf['realised_pl'].sum(), 3)
            fpl = round(ovdf['floating_pl'].sum(), 3)
            current_stake_str = f'### Stake Balance: {total_current_stake:0.3f} NMR'
            float_stake_str =  f'### Floating Balance: {total_floating_stake:0.3f} NMR'
            if rpl>=0:
                real_pl_color = 'green'
            else:
                real_pl_color = 'red'
            if fpl>=0:
                float_pl_color = 'green'
            else:
                float_pl_color = 'red'
            real_pl_str = f'### Realised P/L: <span style="color:{real_pl_color}">{rpl}</span> NMR'
            float_pl_str = f'### Floating P/L: <span style="color:{float_pl_color}">{fpl}</span> NMR'
            stake_cts[0].markdown(current_stake_str, unsafe_allow_html=True)
            stake_cts[1].markdown(float_stake_str, unsafe_allow_html=True)
            pl_cts[0].markdown(real_pl_str, unsafe_allow_html=True)
            pl_cts[1].markdown(float_pl_str, unsafe_allow_html=True)
            date_label.subheader(f'Date: {numerai_date}')
            if st.button('show breakdown by live rounds'):
                liveround_exp = st.expander('',expanded=True)
                with liveround_exp:
                    stake_models = ovdf['model'].tolist()
                    liveround_stake_df = get_stake_by_liverounds(stake_models)
                    # st.write(liveround_stake_df)

                    round_view(liveround_stake_df,'live_round_stake')



def app_setting():
    pfm_exp = st.expander('Perormance Data Setting', expanded=True)
    with pfm_exp:
        pfm_default_model= st.checkbox('download data for default model', value=True)

    stake_exp = st.expander('stake overview data setting', expanded=True)
    if st.button('confirm settiong'):
        st.session_state['pfm_default_model'] = pfm_default_model



def performance_overview():
    select_app = st.sidebar.selectbox("", list(pfm_opt.keys()), index=0, format_func=lambda x: pfm_opt[x])
    if select_app=='data_op':
        data_operation()

    if select_app=='performance_overview':
        performance_overview()
    if select_app=='historic_trend':
        histtrend()
    if select_app=='model_evaluation':
        model_evaluation()



def show_content():
    st.sidebar.header('Dashboard Selection')
    select_app = st.sidebar.selectbox("", list(app_opt.keys()), index=0, format_func=lambda x: app_opt[x])
    if select_app=='performance_overview':
        performance_overview()
    if select_app=='stake_overview':
        stake_overview()
    if select_app=='app_setting':
        app_setting()


# main body
# various configuration setting
app_opt = {
           'performance_overview' : 'Performance Overview',
           'stake_overview': 'Stake Overview',
           'app_setting':''
           }


pfm_opt = {
    'data_op': 'Download Score Data',
    'liveround_view': 'Live Round Overview',
    'historic_trend': 'Historic Trend',
    'model_evaluation': 'Model Evaluation',
}



tbl_opt = {
            'round_result':'Round Results',
            'dailyscore_metric':'Daily Score Metrics',
            'round_metric' : 'Round Metrics'
}

id_metric_opt = {
                'id_corr_sharpe':'Daily Score corr sharpe',
                'id_mmc_sharpe': 'Daily Score mmc sharpe',
                'id_corrmmc_sharpe': 'Daily Score corrmmc sharpe',
                'id_corr2mmc_sharpe': 'Daily Score corr2mmc sharpe',
                'id_corrmmcpct_sharpe': 'Daily Score corrmmc avg pct sharpe',
                'id_corr2mmcpct_sharpe': 'Daily Score corr2mmc avg pct sharpe',
                'id_corrpct_sharpe':'Daily Score corr pct sharpe',
                'id_mmcpct_sharpe': 'Daily Score mmc pct sharpe',
}


id_metric_score_dic = {
                'id_corr_sharpe':'corr',
                'id_mmc_sharpe': 'mmc',
                'id_corrmmc_sharpe': 'corrmmc',
                'id_corr2mmc_sharpe': 'corr2mmc',
                'id_corrmmcpct_sharpe': 'cmavg_pct',
                'id_corr2mmcpct_sharpe': 'c2mavg_pct',
                'id_corrpct_sharpe':'corr_pct',
                'id_mmcpct_sharpe': 'mmc_pct'
}


roundmetric_opt ={'corr':'Corr metrics',
                  'mmc' : 'MMC metrics',
                  'corrmmc' : 'CorrMMC metrics',
                  'corr2mmc' : 'Corr2MMC metrics',
                  'pct' : 'Pecentage metrics'

}


histtrend_opt = {
                'corr':'Correlation',
                'mmc': 'MMC',
                'corrmmc': 'Correlation+MMC',
                'corr2mmc': 'Correlation+2*MMC',
                'corr_pct': 'Correlation Percentile',
                'mmc_pct':'MMC Percentile',
                'cmavg_pct': 'Correlation+MMC Average Percentile',
                'c2mavg_pct': 'Correlation+2*MMC Average Percentile',

}


model_eval_opt = {
        'corr_sharpe' : 'Correlation Sharpe',
        'mmc_sharpe' : 'MMC Sharpe',
        'corrmmc_sharpe' : 'Correlation+MMC Sharpe',
        'corr2mmc_sharpe': 'Correlation+2*MMC Sharpe',
        'corr_mean':'Avg. Correlation',
        'mmc_mean':'Avg. MMC',
        'count': 'Number of Rounds',
        'corrmmc_mean': 'Avg. Correlation+MMC',
        'corr2mmc_mean': 'Avg. Correlation+2*MMC',
        'corr_pct_mean': 'Avg. Correlation Percentile',
        'mmc_pct_mean': 'Avg. MMC Percentile',
        'cmavg_pct_mean': 'Avg. Correlation+MMC Percentile',
        'c2mavg_pct_mean': 'Avg. Correlation+2*MMC Percentile',
        'id_corr_sharpe': 'Daily Score corr sharpe',
        'id_mmc_sharpe': 'Daily Score mmc sharpe',
        'id_corrmmc_sharpe': 'Daily Score corrmmc sharpe',
}

stakeoverview_plot_opt = {
    '1month':'1 Month',
    '3month':'3 Months',
    '1year':'1 Year',
    'all':'Display all available data'
}

def show_session_status_info():
    # 'raw_performance_data'
    key1 = 'model_data'
    if check_session_state(key1) is None:
        st.write(f'{key1} is None')
    else:
        st.write(f'{key1} shape is {st.session_state[key1].shape}')
    pass



project_utils.reload_project()

height_exp = st.sidebar.expander('Plots and tables setting', expanded=False)
with height_exp:
    max_height = st.slider('Please choose the height for plots', 100, 1000, 400, 50)
    max_table_height = st.slider('Please choose the height for tables', 100, 1000, 500, 50)

st.title('Numerai Dashboard')

# key = 'pfm_default_model'
# if check_session_state('pfm_default_model') is None:
#     st.write('set value')
#     st.session_state['pfm_default_model'] = True
# else:
#     st.write('use set value')
#
# st.write(st.session_state)

df = get_saved_data()
show_session_status_info()
# st.write(f'{key} is {chkval}')

# trying out multi columns
# col1, col2 = st.columns(2)
# col1.header('col1')
# col2.header('col2')

show_content()
