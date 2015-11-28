import pandas as pd

from utils.utils import *
from utils.quandl import *


def build_model(clf, df, panel):
    
    df = ts_score(df, panel)

    clf.fit(df[[c for c in df.columns if c != 'tar']], df['tar'])
      
    pred = pd.Series(clf.predict(df[[c for c in df.columns if c != 'tar']]), index=df.index, name='pred')
    sns.jointplot(pred, df['tar'], kind='reg')

    score = clf.score(df[[c for c in df.columns if c != 'tar']], df['tar'])

    ret = load_returns().stack().ix[df.index]
    
    df_res = stack_and_align([df['tar'], pred, ret], cols=('tar', 'pred', 'ret'))
    df_res['err'] = df_res['tar'] - df_res['pred']
    df_res['err2'] = df_res['err'].map(lambda x: x**2)
    avg_tar = df_res['tar'].unstack().mean(axis=1)
    df_res['avg_tar'] = pd.DataFrame({c: avg_tar for c in df_res.index.levels[1]}).stack()

    fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(FIG_WIDTH*2, FIG_HEIGHT*5))
    gen_quintile_flat(df_res, 'tar', 'pred', agg='mean', ts=False).plot(kind='bar', ax=axes[0,0], title='tar vs pred (xs)')
    gen_quintile_flat(df_res, 'tar', 'pred', agg='mean', ts=True).plot(kind='bar', ax=axes[1,0], title='tar vs pred (ts)')
    gen_quintile_flat(df_res, 'avg_tar', 'pred', agg='mean', ts=True).plot(kind='bar', ax=axes[2,0], title='xs avg tar vs pred (ts)')
    gen_quintile_flat(df_res, 'tar', 'err2', agg='sum', ts=False).plot(kind='bar', ax=axes[0,1], title='tar vs err2 (xs)')
    gen_quintile_flat(df_res, 'tar', 'err2', agg='sum', ts=True).plot(kind='bar', ax=axes[1,1], title='tar vs err2 (ts)')
    gen_quintile_flat(df_res, 'avg_tar', 'err2', agg='sum', ts=True).plot(kind='bar', ax=axes[2,1], title='xs avg tar vs err2 (ts)')
    gen_quintile_ts(df_res, 'pred', 'pred', agg='mean').plot(ax=axes[3,0], title='avg pred over time')
    gen_quintile_ts(df_res, 'tar', 'tar', agg='mean').plot(ax=axes[3,1], title='avg tar over time')
    
    avg_rank_accuracy(df_res).plot(ax=axes[4,0], title='avg pred rank accuracy')

    q = gen_quintile_ts(df_res, 'pred', 'ret', agg='mean')
    q['mkt'] = load_returns('states').ix[:,0]
    print('sharpe ratios:')
    print(get_sharpe_ratio(q))
    get_cum_perforance(q).plot(ax=axes[4,1], title='continuously invested performance')
    
    return clf, df_res, score, pred
