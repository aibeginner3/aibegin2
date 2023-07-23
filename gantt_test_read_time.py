import pandas as pd
import plotly.express as px
import plotly.io as pio
import datetime

# データをcsvから入力
#infile = '稼働時間_2023年7月1日昼.csv'

files = ['稼働時間_2023年7月1日昼.csv','稼働時間_2023年7月1日夜.csv','稼働時間_2023年7月2日昼.csv']

data_list=[]

for infile in files:

    # データ読み込み
    df_infile = pd.read_csv(infile, index_col=0).fillna(0)

    # Columns, Index
    Machines = list(df_infile.columns)
    Parts = list(df_infile.index)

    print(df_infile)
    print(Machines)
    print(Parts)

    # 日付読み取り
    target_date = infile.split('.csv')[0].split('_')[1][:-1]
    print(target_date)

    # 昼勤/夜勤
    start_hour=''
    if '昼' in infile:
        start_hour = 9
    elif '夜' in infile:
        start_hour = 21
    print(start_hour)

    timestamp = pd.to_datetime(target_date, format='%Y年%m月%d日')
    start_time = timestamp + datetime.timedelta(hours=start_hour)
    print(timestamp)
    print(start_time)

    # ガントチャート用データ作成
    AM_time = 3  # 始業から昼休憩までの時間設定
    for machine in Machines:
        for part in Parts:
            if df_infile[machine][part] != 0:
                print(part, machine, df_infile[machine][part])
                if df_infile[machine][part]>AM_time:
                    end_time = start_time + datetime.timedelta(hours=AM_time)
                    data_list.append([part, machine, start_time, end_time])
                    start_time2 = start_time + datetime.timedelta(hours=AM_time+1)
                    end_time2   = start_time + datetime.timedelta(hours=df_infile[machine][part]+1)
                    data_list.append([part, machine, start_time2, end_time2])
                else:
                    end_time = start_time + datetime.timedelta(hours=df_infile[machine][part])
                    data_list.append([part, machine, start_time, end_time])

col_data = ['Part','Machine','Start','Finish']
row_data = range(len(data_list))

# ガントチャート用データフレーム
df = pd.DataFrame(data_list, index=row_data, columns=col_data)
print(df)

fig = px.timeline(
    df,  # 使用するデータフレーム
    x_start='Start', x_end='Finish',  # 横軸の開始・終了の列名
    #y='Part',  # 縦軸の列名をリソースに変更
    #color='Machine',  # 色分けをリソースにする
    color='Part',  # 縦軸の列名をリソースに変更
    y='Machine',  # 色分けをリソースにする
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
fig.update_yaxes(categoryarray=Machines, autorange='reversed')
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



