import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.metrics import r2_score

from utils import utils as ut
from utils import quandl as ql
from settings import *


def explore_series(px, px_ca, px_us, tar):
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(FIG_WIDTH*2, FIG_HEIGHT*3))
    px_us.plot(ax=axes[0,0], title='ca and us sig', legend=True)
    px_ca.plot(ax=axes[0,0], legend=True)
    px.plot(ax=axes[0,1], legend=False, alpha=.3)

    (ut.lead_lag_corr(px, tar, rng=range(-52,52,4))
     .plot(kind='bar', title='lead lag corr', ax=axes[1,0]))#.axvline(0, linestyle='--', color='r'))

    df = ut.stack_and_align([px, tar], cols=('sig','tar')).dropna()
    df = ut.ts_score(df)
    sns.distplot(df['sig'], ax=axes[2,0]).set_title('sig dist')
    sns.regplot(df['sig'], df['tar'], ax=axes[2,1]).set_title('sig vs tar')

    clf = lm.LinearRegression()
    clf.fit(df[['sig']], df['tar'])
    score = clf.score(df[['sig']], df['tar'])
    corr = ut.get_xs_corr(df['sig'], df['tar'])

    print('int: {0:03f}\tcoef: {1:03f}\tr2 score: {2:03f}\txs corr: {3:03f}'.format(clf.intercept_, clf.coef_[0], score, corr))


def rolling_fit(clf, df):
    split_date = df.ix[0].name[0] + datetime.timedelta(days=30*12*3)
    end_date = df.ix[-1].name[0]
    inc = datetime.timedelta(days=30*6)
    preds = []
    while split_date < end_date:
        #print(split_date)
        train = df[:split_date].copy(deep=True)
        test = df[split_date:(split_date + inc)].copy(deep=True)
        print("{0}\t{1}\t{2}\t{3}".format(train.iloc[0].name[0],
                                          train.iloc[-1].name[0],
                                          test.iloc[0].name[0],
                                          test.iloc[-1].name[0]))
        clf.fit(train[[c for c in train.columns if c != 'tar']], train['tar'])
        pred = pd.Series(clf.predict(test[[c for c in train.columns if c != 'tar']]), 
                         index=test.index, name='pred')
        preds.append(pred)
        
        split_date += inc 
        if split_date >= end_date:
            break
        
    return pd.concat(preds)


def model_empirics(clf, df, pred):
    
    sns.jointplot(pred, df['tar'], kind='reg')

    score = r2_score(df['tar'], pred)
    corr = ut.get_xs_corr(pred, df['tar'])
    print('r2: {0:03f}\txs corr: {1:03f}'.format(score, corr))

    ret = ql.load_returns().stack().ix[df.index]    
    df_res = ut.stack_and_align([df['tar'], pred, ret], cols=('tar', 'pred', 'ret'))
    df_res['err'] = df_res['tar'] - df_res['pred']
    df_res['err2'] = df_res['err'].map(lambda x: x**2)
    avg_tar = df_res['tar'].unstack().mean(axis=1)
    df_res['avg_tar'] = pd.DataFrame({c: avg_tar for c in df_res.index.levels[1]}).stack()

    fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(FIG_WIDTH*2, FIG_HEIGHT*6))
    (ut.gen_quintile_flat(df_res, 'tar', 'pred', agg='mean', ts=False)
        .plot(kind='bar', ax=axes[0,0], title='tar vs pred (xs)'))
    (ut.gen_quintile_flat(df_res, 'tar', 'pred', agg='mean', ts=True)
        .plot(kind='bar', ax=axes[1,0], title='tar vs pred (ts)'))
    (ut.gen_quintile_flat(df_res, 'avg_tar', 'pred', agg='mean', ts=True)
        .plot(kind='bar', ax=axes[2,0], title='xs avg tar vs pred (ts)'))
    (ut.gen_quintile_flat(df_res, 'tar', 'err2', agg='sum', ts=False)
        .plot(kind='bar', ax=axes[0,1], title='tar vs err2 (xs)'))
    (ut.gen_quintile_flat(df_res, 'tar', 'err2', agg='sum', ts=True)
        .plot(kind='bar', ax=axes[1,1], title='tar vs err2 (ts)'))
    (ut.gen_quintile_flat(df_res, 'avg_tar', 'err2', agg='sum', ts=True)
        .plot(kind='bar', ax=axes[2,1], title='xs avg tar vs err2 (ts)'))
    (ut.gen_quintile_ts(df_res, 'pred', 'pred', agg='mean')
        .plot(ax=axes[3,0], title='avg pred over time'))
    (ut.gen_quintile_ts(df_res, 'tar', 'tar', agg='mean')
        .plot(ax=axes[3,1], title='avg tar over time'))
    
    q = ut.gen_quintile_ts(df_res, 'pred', 'ret', agg='mean')
    q['mkt'] = ql.load_returns('states').ix[:,0]
    print('\n')
    print('sharpe ratios:')
    print(ut.get_sharpe_ratio(q))

    ut.avg_rank_accuracy(df_res).plot(ax=axes[4,0], title='avg pred rank accuracy')
    ut.get_cum_perforance(q).plot(ax=axes[4,1], title='continuously invested performance')
    
    df_res['tar'].unstack().corrwith(df_res['pred'].unstack(), axis=1).plot(ax=axes[5,0], ylim=(-1,1), title='xs tar-pred corr over time')

    return df_res
