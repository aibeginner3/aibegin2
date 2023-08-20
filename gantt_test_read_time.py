# -*- coding: Shift-JIS -*-

import math

# input data
f = open('lambda.txt', 'r')
datalist = f.readlines()
f.close()

mu = 0.10

for line in datalist:
    if 'lambda' in line:
        line2 = line.rstrip('\n').replace('  ', ' ').replace('  ', ' ')
        print(line2)
        (w, w, width, w, lamb, w, w, dos, w, w, omega_log) = line2.split(' ')
        width     = float(width)
        lamb      = float(lamb)
        dos       = float(dos)
        omega_log = float(omega_log)
        print(width, lamb, dos, omega_log)


        Tc = omega_log/1.2*math.exp( (-1.04*(1+lamb)) / (lamb*(1-0.62*mu)-mu) )
        print('Tc= ', Tc)

exit()




# -*- coding: Shift-JIS -*-

from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import pandas as pd
import matplotlib.pyplot as plt
import os, time
import itertools

################################################################################
def plot_num_all(f1, output):

    # ファイル読込
    df_data = pd.read_csv(f1, index_col=0, encoding='cp932')

    # ラベル
    prods   = list(df_data.index.values)
    periods = list(df_data.columns.values)

    # グラフ設定
    #plt.title(f'', fontsize=16, fontname = 'MS Gothic')
    plt.xlabel('期間', fontsize=10, fontname = 'MS Gothic')
    plt.ylabel('生産数', fontsize=10, fontname = 'MS Gothic')
    plt.xticks(rotation=45, fontname = 'MS Gothic')
    plt.ylim(0, 8000)
    plt.grid(True) # 目盛線表示
    plt.tick_params(labelsize=8)

    # グラフ描画
    btm = [0 for p in periods]
    for nprod in prods:
        plt.bar(periods, list(df_data.T[nprod].values), bottom=btm,
            tick_label=periods, align="center", label=nprod)
        for i in range(len(periods)):
            btm[i] += list(df_data.T[nprod].values)[i]
    #plt.legend(loc="upper right", fontsize=4, prop={'family':'MS Gothic'}) # (5)凡例表示
    #plt.show()
    plt.savefig(output)

################################################################################

# input data
df_nums      = pd.read_csv('data\\data_数量.csv', index_col=0, encoding = 'cp932')
df_limits    = pd.read_csv('data\\data_上限稼働時間.csv', index_col=0, encoding = 'cp932')
df_volume    = pd.read_csv('data\\data_出来高.csv', index_col=0, encoding = 'cp932')
df_stock     = pd.read_csv('data\\data_初期在庫.csv', index_col=0, encoding = 'cp932')
df_min_stock = pd.read_csv('data\\data_MIN在庫.csv', index_col=0, encoding = 'cp932')
df_max_stock = pd.read_csv('data\\data_MAX在庫.csv', index_col=0, encoding = 'cp932')
print(df_nums)
print(df_limits)
print(df_volume)
print(df_stock)
print(df_min_stock)
print(df_max_stock)
#exit()

df_vol = 1/df_volume.replace(0, 0.0001)
print(df_vol)

prods         = list(df_nums.index.values)
units         = list(df_volume.columns.values)
periods       = list(df_nums.columns.values)
lbl_limit     = list(df_limits.index.values)[0]
lbl_stock     = list(df_stock.columns.values)[0]
lbl_min_stock = list(df_min_stock.columns.values)[0]
lbl_max_stock = list(df_max_stock.columns.values)[0]

print(prods)
print(units)
print(periods)
print(lbl_limit)
print(lbl_stock)
print(lbl_min_stock)
print(lbl_max_stock)
#exit()

status_all                = []
production_num            = []
production_num_sum        = []
production_time_sum       = []
production_stock_sum      = []
production_stock_sum_diff = []
production_stock_sum_loss = []
check_all                 = []
check_all_must_prod       = []
check_all_must_prod_num   = []
check_all_must_prod_time  = []

production_stock_sum.append(list(df_stock.sum(axis=1)))                # 初期在庫
df_volume_median = df_volume.sum(axis=1)/(df_volume > 0).sum(axis=1)   # 平均値
df_volume_min = df_volume.replace(0, 999).min(axis=1)                  # 最小値
#print(df_volume_median)
#print(df_volume_min)
#input()

# output
nowtime = time.strftime('%Y%m%d%H%M%S', time.localtime())
outdir  = 'result_' + nowtime
outdir_nums  = outdir + '\\品番別生産数'
outdir_time  = outdir + '\\号機別稼働時間'
outdir_check = outdir + '\\チェックログ'

os.mkdir(outdir)
os.mkdir(outdir_nums)
os.mkdir(outdir_time)
os.mkdir(outdir_check)

fname_result_status_all                = outdir + '\\result_status.csv'
fname_result_production_num            = outdir + '\\result_production_num.csv'
fname_result_production_num_sum        = outdir + '\\result_production_num_sum.csv'
fname_result_production_time_sum       = outdir + '\\result_production_time_sum.csv'
fname_result_production_stock_sum      = outdir + '\\result_production_stock_sum.csv'
fname_result_production_stock_sum_diff = outdir + '\\result_production_stock_sum_diff.csv'
fname_result_production_stock_sum_loss = outdir + '\\result_production_stock_sum_loss.csv'
fname_result_check_all                 = outdir + '\\result_check.csv'
fname_result_check_must_prod           = outdir + '\\result_check_must_prod.csv'
fname_result_check_must_prod_num       = outdir + '\\result_check_must_prod_num.csv'
fname_result_check_must_prod_time      = outdir + '\\result_check_must_prod_time.csv'

outfig = outdir + '\\plot_num.png'

# 生産計画
for ymd in periods:

    print('########################################')
    print(ymd)

    # 在庫日数, MAX在庫との差分
    df_stockday    = df_stock.copy()
    df_stockdif    = df_stock.copy()
    df_stocklos    = df_stock.copy()
    must_prod      = []
    must_prod_num  = []
    must_prod_time = []
    for nprod in prods:
        if df_nums[ymd][nprod]==0:
            df_stockday[lbl_stock][nprod] = 999
        else:
            df_stockday[lbl_stock][nprod] = df_stock[lbl_stock][nprod]/df_nums[ymd][nprod]

        df_stockdif[lbl_stock][nprod] = df_max_stock[lbl_max_stock][nprod] - df_stock[lbl_stock][nprod]
        df_stocklos[lbl_stock][nprod] = df_stock[lbl_stock][nprod] - df_nums[ymd][nprod] - df_min_stock[lbl_min_stock][nprod]
        if df_stocklos[lbl_stock][nprod]<0 and df_nums.T[nprod].sum()>0:
            print('### ', nprod)
            must_prod.append(nprod)
            must_prod_num.append(df_stocklos[lbl_stock][nprod])
            must_prod_time.append(df_stocklos[lbl_stock][nprod]/df_volume_min[nprod])

    if len(must_prod)<5:
        if df_nums[ymd].sum()==0:
            df_tmp_stockdif = df_stockdif.copy()
            for i in range(5):
                nprod = df_tmp_stockdif[lbl_stock].idxmax()
                must_prod.append(nprod)
                must_prod_num.append(df_stocklos[lbl_stock][nprod])
                must_prod_time.append(df_stocklos[lbl_stock][nprod]/df_volume_min[nprod])
                df_tmp_stockdif[lbl_stock][nprod]=0

        else:
            df_tmp_stockday = df_stockday.copy()
            for i in range(5):
                nprod = df_tmp_stockday[lbl_stock].idxmin()
                must_prod.append(nprod)
                must_prod_num.append(df_stocklos[lbl_stock][nprod])
                must_prod_time.append(df_stocklos[lbl_stock][nprod]/df_volume_min[nprod])
                df_tmp_stockday[lbl_stock][nprod]=99999

    check_all_must_prod.append(must_prod)
    check_all_must_prod_num.append(must_prod_num)
    check_all_must_prod_time.append(must_prod_time)

    print(df_stockday)
    print(df_stockdif)
    print(must_prod, len(must_prod))

    ## 出荷がある品番の抽出
    #select_nums_prod=[]
    #if df_nums[ymd].sum()==0:
    #    for nprod in prods:
    #        diff    = df_max_stock[lbl_max_stock][nprod] - df_stock[lbl_stock][nprod]
    #        sum_num = df_nums.T[nprod].sum()
    #        if diff>0 and sum_num>0:
    #            select_nums_prod.append(nprod)
    #else:
    #    for nprod in prods:
    #        #stock_num = df_stock[lbl_stock][nprod] - df_nums[ymd][nprod]
    #        if df_nums[ymd][nprod]>0:
    #            select_nums_prod.append(nprod)
    #print(select_nums_prod)

    min_stockday       = []
    select_nums_prod   = []

    # 在庫日数が小さい品番から抽出
    for i in range(len(units)-3):
    #for i in range(len(prods)):
        if df_nums[ymd].sum()==0:
            tmp_prod = df_stockdif[lbl_stock].idxmax()
            if df_stockdif[lbl_stock][tmp_prod]>0 and df_nums.T[tmp_prod].sum()>0:
                select_nums_prod.append(tmp_prod)
                df_stockdif[lbl_stock][tmp_prod]=0
        else:
            tmp_prod = df_stockday[lbl_stock].idxmin()
            if df_stockday[lbl_stock][tmp_prod]<999:
            #if df_stockday[lbl_stock][tmp_prod]<3:
                min_stockday = [tmp_prod, df_stockday[lbl_stock][tmp_prod]]
                select_nums_prod.append(tmp_prod)
                df_stockday[lbl_stock][tmp_prod]=9999

    print(select_nums_prod)
    #input()

    # 出来高表の組み合わせ総当たり
    select_volume=[]
    combination=[]
    for unit in units:
        tmp=[]
        tmp_comb=[]
        for i in range(len(select_nums_prod)):
            tmp.append(df_volume[unit][select_nums_prod[i]])
            if df_volume[unit][select_nums_prod[i]]>0:
                tmp_comb.append(i)
        select_volume.append(tmp)
        if tmp_comb==[]:
            tmp_comb.append(999)
        combination.append(tmp_comb)
    print(select_volume)
    print(combination)

    c1  = combination[0]
    c2  = combination[1]
    c3  = combination[2]
    c4  = combination[3]
    c5  = combination[4]
    c6  = combination[5]
    c7  = combination[6]
    c8  = combination[7]
    c9  = combination[8]
    c10 = combination[9]
    c11 = combination[10]
    c12 = combination[11]
    c13 = combination[12]
    c14 = combination[13]

    if len(must_prod)<10:
        all_combinations = list(itertools.product(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14))
    else:
        all_combinations = list(itertools.product(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14))
    #print(all_combinations)
    print(len(all_combinations))
    input()

    # 最低限必要な品番が含まれている組合せを抽出
    pickup_comb = []
    if len(must_prod)>0:
        must_prod_num=[]
        for i in range(len(select_nums_prod)):
            if select_nums_prod[i] in must_prod:
                must_prod_num.append(i)
        print(must_prod_num)

        print('##### ', len(set(must_prod_num)), len(set(all_combinations[0])))
        if len(set(must_prod_num)) < 10:
            print('#check1 ', len(set(must_prod_num)))
            for comb in all_combinations:
                #print(list(comb), must_prod_num, set(comb), set(must_prod_num), set(list(comb)))
                if set(must_prod_num) <= set(comb):
                        pickup_comb.append(comb)
        else:
            print('#check2 ', len(set(must_prod_num)))
            for comb in all_combinations:
                if set(must_prod_num) >= set(comb) or set(must_prod_num) <= set(comb):
                    if len(set(comb)) >= 9:
                        pickup_comb.append(comb)
    print(len(pickup_comb))


    if pickup_comb==[]:
        pickup_comb = all_combinations.copy()
    print(len(pickup_comb))

    input()

    #for i in range(len(all_combinations)):
    #    print(i, list(all_combinations[i]))

    #print(select_nums_prod)
    #input()

    check_all_ymd=[]
    check_min=0
    best_min_num=0
    best_min_num_stock=-99999999
    best_sum_num_stock_maxdiff=99999999
    for i_comb in range(len(pickup_comb)):

        if i_comb/1000==int(i_comb/1000):
            print(i_comb, comb, len(set(must_prod_num)), len(set(comb)))

        if len(must_prod)<10:
            comb  = pickup_comb[i_comb]
        else:
            comb  = pickup_comb[i_comb][0:len(units)]
            comb2 = pickup_comb[i_comb][len(units):2*len(units)]
        #print(i_comb, comb)
        #input()

        if len(must_prod)<10:
            list_prod=[]
            for i in range(len(comb)):
                #print(comb[i], select_nums_prod[comb[i]], units[i], df_volume[units[i]][select_nums_prod[comb[i]]], df_limits[units[i]][lbl_limit])
                tmp=[]
                for j in range(len(prods)):
                    if comb[i]<99:
                        if select_nums_prod[comb[i]]==prods[j]:
                            tmp.append(int(df_volume[units[i]][prods[j]] * df_limits[units[i]][lbl_limit]))
                        else:
                            tmp.append(0)
                    else:
                        tmp.append(0)
                list_prod.append(tmp)

            df_prod = pd.DataFrame(list_prod, index=units, columns=prods).T

        else:
            list_prod1=[]
            for i in range(len(comb)):
                #print(comb[i], select_nums_prod[comb[i]], units[i], df_volume[units[i]][select_nums_prod[comb[i]]], df_limits[units[i]][lbl_limit])
                tmp=[]
                for j in range(len(prods)):
                    if comb[i]<99:
                        if select_nums_prod[comb[i]]==prods[j]:
                            tmp.append(int(df_volume[units[i]][prods[j]] * df_limits[units[i]][lbl_limit] / 2))
                        else:
                            tmp.append(0)
                    else:
                        tmp.append(0)
                list_prod1.append(tmp)

            list_prod2=[]
            for i in range(len(comb2)):
                #print(comb[i], select_nums_prod[comb[i]], units[i], df_volume[units[i]][select_nums_prod[comb[i]]], df_limits[units[i]][lbl_limit])
                tmp=[]
                for j in range(len(prods)):
                    if comb[i]<99:
                        if select_nums_prod[comb2[i]]==prods[j]:
                            tmp.append(int(df_volume[units[i]][prods[j]] * df_limits[units[i]][lbl_limit] / 2))
                        else:
                            tmp.append(0)
                    else:
                        tmp.append(0)
                list_prod2.append(tmp)

            df_prod1 = pd.DataFrame(list_prod1, index=units, columns=prods).T
            df_prod2 = pd.DataFrame(list_prod2, index=units, columns=prods).T
            df_prod  = df_prod1 + df_prod2

        #print(df_prod)
        #input()

        check_stock       = []
        num_stock         = []
        num_stock_maxdiff = []
        for nprod in select_nums_prod:
            today_prod = df_prod.T[nprod].sum()
            next_stock = df_stock[lbl_stock][nprod] + today_prod - df_nums[ymd][nprod]

            # 生産数の超過確認
            if next_stock > df_max_stock[lbl_max_stock][nprod]:
                #print('MAX在庫超過')
                diff = next_stock - df_max_stock[lbl_max_stock][nprod]
                count=0
                sum_volume=0
                tmp_units=[]
                for tmp_unit in units:
                    if df_prod[tmp_unit][nprod]>0:
                        count += 1
                        sum_volume += df_volume[tmp_unit][nprod]
                        tmp_units.append(tmp_unit)
                if count==1:
                    df_prod[tmp_units[0]][nprod] = int(df_prod[tmp_units[0]][nprod] - diff)
                else:
                    for tmp_unit in tmp_units:
                        df_prod[tmp_unit][nprod] = int(df_prod[tmp_unit][nprod] - diff * df_volume[tmp_unit][nprod] / sum_volume)
                next_stock = df_max_stock[lbl_max_stock][nprod]

            # MIN在庫以上の判定
            num_stock.append(next_stock)
            if next_stock >= df_min_stock[lbl_min_stock][nprod]:
                check_stock.append(1)
            else:
                check_stock.append(0)

            #df_next_stockdiff = df_max_stock[lbl_max_stock][nprod] - next_stock
            #df_next_stockloss = next_stock - df_nums[ymd][nprod] - df_min_stock[lbl_min_stock][nprod]

            # MAX在庫との差分
            num_stock_maxdiff.append(df_max_stock[lbl_max_stock][nprod] - next_stock)

        min_num_stock         = min(num_stock)
        sum_num               = df_prod.sum().sum()
        sum_num_stock_maxdiff = sum(num_stock_maxdiff)

        check_all_ymd.append([len(all_combinations), len(pickup_comb), i_comb] + select_nums_prod + list(comb) + [min_num_stock, sum_num, sum_num_stock_maxdiff] + check_stock + num_stock)
        #print([len(all_combinations), len(pickup_comb), i_comb] + list(comb) + [min_num_stock, sum_num, sum_num_stock_maxdiff] + check_stock + num_stock)

        if sum(check_stock)>check_min and sum_num>best_min_num and min_num_stock>best_min_num_stock and sum_num_stock_maxdiff<best_sum_num_stock_maxdiff:
            best_check_all = [len(all_combinations), len(pickup_comb), i_comb] + select_nums_prod + list(comb) + [min_num_stock, sum_num, sum_num_stock_maxdiff] + check_stock + num_stock
            df_best_prod               = df_prod.copy()
            check_min                  = sum(check_stock)
            best_min_num               = sum_num
            best_min_num_stock         = min_num_stock
            best_sum_num_stock_maxdiff = sum_num_stock_maxdiff

        #input()

    df_check_all_ymd = pd.DataFrame(check_all_ymd)
    df_check_all_ymd.to_csv(outdir_check + '\\result_check_all_{}.csv'.format(ymd), encoding = 'cp932')
    check_all.append(best_check_all)

    print(select_nums_prod)
    print(best_check_all)
    print(df_best_prod)
    #input()

#    # variables
#    #production = LpVariable.dicts('production', (units, select_nums_prod), 0, max(df_nums[ymd]), 'Integer')
#    production = LpVariable.dicts('production', (units, select_nums_prod), lowBound=0, cat='Integer')
#
#    # objective function
#    total_prod = lpSum(production[unit][nprod] for unit in units for nprod in select_nums_prod if df_volume[unit][nprod]>=1)
#
#    # model and objectives
#    model = LpProblem("Manufacture", LpMaximize)
#    model += total_prod
#
#    # constraints
#    ### sum of production
#    #for nprod in select_nums_prod:
#    #    total_prod_amount = lpSum(produce[unit][nprod] for unit in units)
#    #    model += total_prod_amount <= df_needs['needs'][nprod]
#
#    ## upper limits
#    for unit in units:
#        total_time_each_unit = lpSum(production[unit][nprod] * df_vol[unit][nprod] for nprod in select_nums_prod)
#        model += total_time_each_unit <= df_limits[unit][lbl_limit]
#
#    ### variables
#    #for unit in units:
#    #    for nprod in select_nums_prod:
#    #        if df_volume[unit][nprod]==0:
#    #            model += production[unit][nprod]==0
#
#    ## stock
#    for nprod in select_nums_prod:
#        today_prod = lpSum(production[unit][nprod] for unit in units)
#        next_stock = df_stock[lbl_stock][nprod] + today_prod - df_nums[ymd][nprod]
#        #model += next_stock >= 0
#        model += next_stock >= df_min_stock[lbl_min_stock][nprod]
#        model += next_stock <= df_max_stock[lbl_max_stock][nprod]
#
#    status = model.solve(PULP_CBC_CMD( timeLimit=2 ))
#    status_all.append(LpStatus[status])
#
#    print("Status:", LpStatus[status])
#    print(model.objective.value())

    status_all.append('総当たり')

    #print(model)

    print('Result:')

    # 品番別生産数
    #production_value = {nprod: {unit: production[unit][nprod].value() for unit in units} for nprod in prods}
    #df_prod = pd.DataFrame(production_value, index=units, columns=prods).T

    #df_prod = pd.DataFrame([[0 for i in range(len(prods))] for j in range(len(units))], index=units, columns=prods).T
    #for unit in units:
    #    for nprod in select_nums_prod:
    #        df_prod[unit][nprod] = production[unit][nprod].value()
    df_prod = df_best_prod.copy()
    df_prod.to_csv(outdir_nums + '\\result_production_{}.csv'.format(ymd), encoding = 'cp932')
    print(df_prod)

    # 号機別稼働時間
    df_time = df_prod * df_vol.loc[prods]
    #for unit in units:
    #    for nprod in select_nums_prod:
    #        df_time[unit][nprod] = df_prod[unit][nprod] * df_vol[unit][nprod]
    df_time.to_csv(outdir_time + '\\result_production_time_{}.csv'.format(ymd), encoding = 'cp932')
    print(df_time)
    #input()

    print('Number of Production')
    print(df_prod.sum(axis=1))

    print('Production Time')
    print(df_time.sum(axis=0))

    ## update stock
    for nprod in prods:
        today_prod=0
        for unit in units:
            today_prod += df_prod[unit][nprod]
        next_stock = df_stock[lbl_stock][nprod] + today_prod - df_nums[ymd][nprod]
        df_stock[lbl_stock][nprod] = next_stock

    print(df_stock)

    # 品番別号機別生産数
    production_num = production_num + df_prod.T.values.tolist()

    # 品番別合計生産数
    production_num_sum.append(list(df_prod.sum(axis=1)))
    print(production_num_sum)
    #input()

    # 号機別合計稼働時間
    production_time_sum.append(list(df_time.sum(axis=0)))
    print(production_time_sum)
    #input()

    # 品番別合計在庫数
    production_stock_sum.append(list(df_stock.sum(axis=1)))

    # 品番別在庫数対MAX在庫差分
    production_stock_sum_diff.append(list(df_stockdif.sum(axis=1)))

    # 品番別在庫不足数
    production_stock_sum_loss.append(list(df_stocklos.sum(axis=1)))

    #input()

print('###############')
print('### summary ###')
print('###############')

df_status_all = pd.DataFrame(status_all, index=periods, columns=['status']).T
print(df_status_all)
df_status_all.to_csv(fname_result_status_all, encoding = 'cp932')

df_production_num_sum = pd.DataFrame(production_num_sum, index=periods, columns=prods).T
print(df_production_num_sum)
df_production_num_sum.to_csv(fname_result_production_num_sum, encoding = 'cp932')

df_production_time_sum = pd.DataFrame(production_time_sum, index=periods, columns=units).T
print(df_production_time_sum)
df_production_time_sum.to_csv(fname_result_production_time_sum, encoding = 'cp932')

df_production_stock_sum = pd.DataFrame(production_stock_sum, index=['初期在庫'] + periods, columns=prods).T
print(df_production_stock_sum)
df_production_stock_sum.to_csv(fname_result_production_stock_sum, encoding = 'cp932')

df_production_stock_sum_diff = pd.DataFrame(production_stock_sum_diff, index=periods, columns=prods).T
df_production_stock_sum_diff.to_csv(fname_result_production_stock_sum_diff, encoding = 'cp932')

df_production_stock_sum_loss = pd.DataFrame(production_stock_sum_loss, index=periods, columns=prods).T
df_production_stock_sum_loss.to_csv(fname_result_production_stock_sum_loss, encoding = 'cp932')

df_check_all = pd.DataFrame(check_all, index=periods)
df_check_all.to_csv(fname_result_check_all, encoding = 'cp932')
#print(df_check_all)

df_check_must_prod = pd.DataFrame(check_all_must_prod, index=periods)
df_check_must_prod.to_csv(fname_result_check_must_prod, encoding = 'cp932')

df_check_must_prod_num = pd.DataFrame(check_all_must_prod_num, index=periods)
df_check_must_prod_num.to_csv(fname_result_check_must_prod_num, encoding = 'cp932')

df_check_must_prod_time = pd.DataFrame(check_all_must_prod_time, index=periods)
df_check_must_prod_time.to_csv(fname_result_check_must_prod_time, encoding = 'cp932')

#print(status_all)
#print(production_num)
#print(production_num_sum)
#print(production_time_sum)
#print(production_stock_sum)

# グラフ
plot_num_all(fname_result_production_num_sum, outfig)



exit()






