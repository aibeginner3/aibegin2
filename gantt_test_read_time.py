test

# -*- coding: Shift-JIS -*-

import pandas as pd
import os, datetime
import plotly.express as px
import plotly.io as pio

# input data
df_data0 = pd.read_csv('data\\data.csv', index_col=0, encoding = 'cp932')

prods = list(df_data0.index.values)
units = list(df_data0.columns.values)
print(prods)
print(units)

dir_result = 'result_dir'

data_list=[]

for unit in units:

    result_filename = dir_result + '/result_' + unit + '.csv'
    print(result_filename)

    df_result = pd.read_csv(result_filename, index_col=0, encoding = 'cp932')
    #print(df_result)

    periods = list(df_result.columns.values)
    #print(periods)

    for period in periods:

        # 日付読み取り
        target_date = period[:-1]
        #print(target_date)

        # 昼/夜
        start_hour=''
        if '昼' in period:
            start_hour = 9
        elif '夜' in period:
            start_hour = 21
        #print(start_hour)

        #timestamp = pd.to_datetime(target_date, format='%Y年%m月%d日')
        timestamp = pd.to_datetime(target_date, format='%m月%d日')
        start_time = timestamp + datetime.timedelta(hours=start_hour)
        #print(timestamp)
        #print(start_time)

        # ガントチャート用データ作成
        start_time0 = start_time
        rest_time = 3.5  # 始業から昼休憩までの時間設定
        time1 = 0
        time2 = 0
        for nprod in prods:
            if df_result[period][nprod] != 0:
                time2 += df_result[period][nprod]
                #print(period, nprod, unit, df_result[period][nprod])
                if time1<rest_time and time2<=rest_time:
                    end_time = start_time + datetime.timedelta(hours=float(df_result[period][nprod]))
                    data_list.append([nprod, unit, start_time, end_time])
                    time1 = time2
                    start_time = end_time
                elif time1<rest_time and rest_time<time2:
                    end_time = start_time0 + datetime.timedelta(hours=rest_time)
                    data_list.append([nprod, unit, start_time, end_time])
                    start_time2 = start_time0 + datetime.timedelta(hours=rest_time+1)
                    end_time2   = start_time + datetime.timedelta(hours=float(df_result[period][nprod])+1)
                    data_list.append([nprod, unit, start_time2, end_time2])
                    time1 = time2
                    start_time = end_time2
                elif rest_time==time1 and rest_time<time2:
                    start_time = start_time + datetime.timedelta(hours=1)
                    end_time = start_time + datetime.timedelta(hours=float(df_result[period][nprod]))
                    data_list.append([nprod, unit, start_time, end_time])
                    time1 = time2
                    start_time = end_time
                elif rest_time<time1 and rest_time<time2:
                    end_time = start_time + datetime.timedelta(hours=float(df_result[period][nprod]))
                    data_list.append([nprod, unit, start_time, end_time])
                    time1 = time2
                    start_time = end_time

    #print(pd.DataFrame(data_list, columns=['品','号','開始時間','終了時間']))
    #input()

col_data = ['品','号','開始時間','終了時間']
row_data = range(len(data_list))

# ガントチャート用データフレーム
df = pd.DataFrame(data_list, index=row_data, columns=col_data)
output_gantt_chart_csv = 'data_gantt_chart.csv'
df.to_csv(output_gantt_chart_csv, encoding = 'cp932')
print(df)

fig = px.timeline(
    df,  # 使用するデータフレーム
    x_start='開始時間', x_end='終了時間',  # 横軸の開始・終了の列名
    #y='品',  # 縦軸の列名をリソースに変更
    #color='号',  # 色分けをリソースにする
    color='品',  # 縦軸の列名をリソースに変更
    y='号',  # 色分けをリソースにする
)
# グラフ全体とホバーのフォントサイズ変更
fig.update_layout(font_size=16, hoverlabel_font_size=16)

#fig.update_layout(plot_bgcolor="white")
fig.update_layout(plot_bgcolor="lightyellow")
fig.update_xaxes(linecolor='black', gridcolor='gray',mirror=True)
fig.update_yaxes(linecolor='black', gridcolor='gray',mirror=True)

# 横軸のレイアウト修正
fig.update_xaxes(
    # 横軸の書式を変更
    #tickformat='%Y年%m月%d日',
    #tickformat='%H:%M',
    tickformat='%m月%d日\n %H:%M',

    # 目盛の範囲
    #range=(datetime.date(2023, 7, 1), datetime.date(2023, 7, 2)),

    # 目盛の間隔
    #dtick='D1',
    
    # グラフ上にレンジセレクターを追加
    rangeselector=dict(
        buttons=list([
            # 1日間
            dict(count=1, label='1day', step='day', stepmode='backward'),
            # 3日間
            dict(count=3, label='3day', step='day', stepmode='backward'),
            # 7日間=1週間
            dict(count=7, label='1week', step='day', stepmode='backward'),
            # 1ヶ月
            dict(count=1, label='1month', step='month', stepmode='backward'),
            # 全期間
            dict(step='all')
        ])
    ),
    # グラフ下にレンジスライダーを追加
    rangeslider=dict(visible=True),
)

# 縦軸の向きを逆にする
#fig.update_yaxes(autorange='reversed')
fig.update_yaxes(categoryarray=units, autorange='reversed')
fig.update_layout(legend=dict(traceorder='normal'))

# グラフ保存
prefix = 'plotly-gantt-charts_test_稼働時間'  # 保存ファイル名の接頭辞
save_name = f"{prefix}_test1"
pio.write_html(fig, f"{save_name}.html")

exit()

#fig.show()


# レイアウトの修正
#fig.update_layout(title=dict(text='<b>Time Series of Stock Prices',
#                             font=dict(size=26,
#                                       color='grey'),
#                             y=0.88,
#                            ),
#                  legend=dict(xanchor='left',
#                              yanchor='bottom',
#                              x=0.02,
#                              y=0.78,
#                             ),
#                  xaxis=dict(title='2020',
#                             tickformat='%b',
#                             range=(datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)),
#                             dtick='M1'),
#                  yaxis=dict(title='normalized stock price',
#                             dtick=100)
#                   
#)



exit()




