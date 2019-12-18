#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import bisect
import pandas as pd
import datetime
from geopy import distance
import os
import platform



    
def find_grid(location):
    return [bisect.bisect(lats,location[0]),bisect.bisect(lngs,location[1])]


def get_sequence(bill,mon,day):
    sequence=[]
    onbus=False
    valid=True
    for i in range(len(bill)):
        item=bill.iloc[i]
        if (item['type']==31):
            plate=item['station_id']
            try:
                bus_gps=pd.read_csv('/home/XXX/Python_Output/Stop_Mapping/2016'+mon+'/'+day+'/modified/'+plate+'.csv',
                                    index_col=0,encoding='utf-8')
            except:
                continue
            location=get_location(item['record_id'],bus_gps)
            if location['latitude']>22 and location['latitude']<25:
                index_i1=bisect.bisect(lats,location['latitude'])
                index_j1=bisect.bisect(lngs,location['longitude'])
                grid_id=point_to_int(index_i1,index_j1,len(grids))
            else:
                valid=False
                break
            if onbus:
                end_grid_id,end_time=get_destination(mon,day,location,sequence[-1]['record_id'],sequence[-1]['key'],sequence[-1]['time'])
                if end_grid_id!=-1:
                    sequence.append({'record_id':sequence[-1]['record_id'],'route':sequence[-1]['route'],'type':'bus_off','key':sequence[-1]['key'],'time':end_time,'grid_id':end_grid_id})
                else:
                    return []
                sequence.append({'record_id':item['record_id'],'route':item['station_name'],'type':'bus_on','key':item['station_id'],'time':item['trans_time'],'grid_id':grid_id})
            else:
                onbus=True
                sequence.append({'record_id':item['record_id'],'route':item['station_name'],'type':'bus_on','key':item['station_id'],'time':item['trans_time'],'grid_id':grid_id})
        elif (item['type']==21):
            if item['station_name'] in stations:
                location=stations[item['station_name']]
                index_i1=bisect.bisect(lats,location['latitude'])
                index_j1=bisect.bisect(lngs,location['longitude'])
                grid_id=point_to_int(index_i1,index_j1,len(grids))
            else:
                valid=False
                break
            if onbus:
                onbus=False
                end_grid_id,end_time=get_destination(mon,day,location,sequence[-1]['record_id'],sequence[-1]['key'],sequence[-1]['time'])
                if end_grid_id!=-1:
                    sequence.append({'record_id':sequence[-1]['record_id'],'route':sequence[-1]['route'],'type':'bus_off','key':sequence[-1]['key'],'time':end_time,'grid_id':end_grid_id})
                else:
                    return []
                sequence.append({'record_id':item['record_id'],'route':item['station_name'],'type':'subway_on','key':item['station_name'],'time':item['trans_time'],'grid_id':grid_id})
            else:
                sequence.append({'record_id':item['record_id'],'route':item['station_name'],'type':'subway_on','key':item['station_name'],'time':item['trans_time'],'grid_id':grid_id})
        elif (item['type']==22):
            if item['station_name'] in stations:
                location=stations[item['station_name']]
                index_i1=bisect.bisect(lats,location['latitude'])
                index_j1=bisect.bisect(lngs,location['longitude'])
                grid_id=point_to_int(index_i1,index_j1,len(grids))
            else:
                valid=False
                break
            sequence.append({'record_id':item['record_id'],'route':item['station_name'],
                             'type':'subway_off','key':item['station_name'],'time':item['trans_time'],'grid_id':grid_id})
    if valid:
        return sequence
    else:
        return []
    
def get_destination(mon,day,location,record_id,car_id,time,final=False):
    if final:
        deadline='2016-'+mon+'-'+day+' 10:00:00'
    else:
        deadline=time
    if location['latitude']>22 and location['latitude']<25:
        try:
            bus_gps=pd.read_csv('/home/XXX/Python_Output/Stop_Mapping/2016'+mon+'/'+day+'/modified/'+car_id+'.csv',index_col=0,encoding='utf-8')
        except:
            return (-1,'')
        point=(location['latitude'],location['longitude'])
        record_i=list(bus_gps['record_id']).index(record_id)
        looper=record_i+1
        last_re=bus_gps.iloc[record_i]
        sign=True
        min_dist=100000
        min_di=None
        while sign:
            if looper>=len(bus_gps):
                sign=False
                continue
            this_re=bus_gps.iloc[looper]
            if this_re['stop_time']==last_re['stop_time']:
                looper+=1
                continue
            p1=(this_re['latitude'],this_re['longitude'])
            dist=distance.great_circle(p1,point).meters
            if dist<min_dist:
                min_dist=dist
                min_di=this_re
            if min_dist<50:
                sign=False
                continue
            if this_re['stop_time']>deadline:
                sign=False
            looper+=1
            last_re=this_re
        if type(min_di)!=type(None):
            index_i1=bisect.bisect(lats,min_di['latitude'])
            index_j1=bisect.bisect(lngs,min_di['longitude'])
            grid_id=point_to_int(index_i1,index_j1,len(grids))
            return (grid_id,min_di['stop_time'])
        else:
            return(-1,'')
    return (-1,'')



def get_location(record,gps):
    try:
        item=gps.loc[lambda d:d['record_id']==record].iloc[0]
    except:
        print(gps)
        raise(ValueError)
    return {'latitude':item['latitude'],'longitude':item['longitude']}


def get_bus(card,morning):
    data=morning.loc[lambda df:(df.card_id==card)]
    if len(data)>0:
        item=data.iloc[0]
        if item['type']==21:
            return {'station':item['station_name'],'card':card,'time':item['trans_time']}
        elif item['type']==31:
            record_id=item['record_id']
            car_id=item['station_id']
            return {'station':car_id,'card':card,'id':record_id}
    return 'nodata'


def fill_station(bill,station_prefix):
    equi_ids=list(bill['equi_id'])
    station_names=list(bill['station_name'])
    station=[]
    for i in range(len(equi_ids)):
        aid=equi_ids[i]
        if station_names[i]!='None':
            station.append(station_names[i])
        elif str(aid)[:6] not in station_prefix:
            station.append(station_names[i])
        else:
            station.append(station_prefix[str(aid//1000)])
    return station



def point_to_int(i,j,row_length):
    return j*row_length+i
def int_to_point(number,row_length):
    return (number%row_length,number//row_length)
        
    
    
def get_agent_m(mon,day,i,source,des):
    grid_info=json.load(open('/home/XXX/Python_Output/grids.json'))
    grids=grid_info['grids']
    cards=json.load(open('/home/XXX/Python_Output/card/'+mon+day+'located_cards_morning.json'))[0][1]
    i=0

    station_prefix=json.load(open('/home/XXX/Python_Output/equipment_prefix_station.json'))
    cleaned_cards=[]
    for card in cards:
        if card['latitude']<22 or card['latitude']>24 or card['longitude']<110 or card['longitude']>116:
            i+=1
        else:
            cleaned_cards.append(card)
    for card in cleaned_cards:
        try:
            grids[bisect.bisect(lats,card['latitude'])][bisect.bisect(lngs,card['longitude'])]['count']+=1
            grids[bisect.bisect(lats,card['latitude'])][bisect.bisect(lngs,card['longitude'])]['cards'].append(card)
        except:
            continue
    g1=int_to_point(source,len(grids))
    g2=int_to_point(des,len(grids))
    cards_matters=[]
    cards_matters.extend(grids[g1[0]][g1[1]]['cards'])
    cards_matters.extend(grids[g2[0]][g2[1]]['cards'])
    print('#candidate cards',len(cards_matters),i,mon,day)
    station_prefix=json.load(open('/home/XXX/Python_Output/equipment_prefix_station.json'))
    bill_index=['record_id','card_id','equi_id','corp_id','type', \
                 'money_nonsense','balance_nonsense','whatever','trans_time','success_sign', \
                'time_1','time_2','corp_name','station_name','station_id']
    df_raw=pd.read_csv('/urbanshare/XXX/Data/12_Billing_szt_SZ/20167-12/P_GJGD_SZT_2016'+mon+day,
                       names=bill_index,encoding='utf-8') \
                       [['record_id','card_id','equi_id','type','money_nonsense','trans_time','station_name','station_id']]
    sts=fill_station(df_raw,station_prefix)
    df_raw['station_name']=sts
    morning=df_raw.loc[lambda df:('2016-'+mon+'-'+day+' 06:00:00'<df.trans_time) & \
                       ('2016-'+mon+'-'+day+' 09:00:00'>df.trans_time)].sort_values('trans_time')
    evening=df_raw.loc[lambda df:('2016-'+mon+'-'+day+' 16:00:00'<df.trans_time) & \
                       ('2016-'+mon+'-'+day+' 23:00:00'>df.trans_time)].sort_values('trans_time')
    sequences=[]
    show=set()
    processed=0
    for card_item in cards_matters:
        card=card_item['card']
        bus=get_bus(card,morning)
        if bus=='nodata':
            continue
        else:
            plate=bus['station']
            if plate in stations:
                location=stations[plate]
            else:
                try:
                    agent_path_morning='/home/XXX/Python_Output/Stop_Mapping/2016'+mon+'/'+day+'/modified/'+plate+'.csv'
                    bus_gps=pd.read_csv(agent_path_morning,
                                        index_col=0,encoding='utf-8')

                except:
                    #print(agent_path_morning,'morning')
                    continue
                location=get_location(bus['id'],bus_gps) 
            if location['latitude']>22 and location['latitude']<25:
                index_i1=bisect.bisect(lats,location['latitude'])
                index_j1=bisect.bisect(lngs,location['longitude'])
                card_bill=morning.loc[lambda df:df.card_id==card]
                sequence=get_sequence(card_bill,mon,day)

                if len(sequence)>0:
                    if sequence[-1]['type']=='bus_on':
                        bus=get_bus(card,evening)
                        if bus=='nodata':
                            continue
                        else:
                            try:
                                plate=bus['station']
                            except:
                                #print(bus['station'])
                                raise(ValueError)
                            if plate in stations:
                                location=stations[plate]
                            else:
                                agent_load_path='/home/XXX/Python_Output/Stop_Mapping/2016'+ \
                                                    mon+'/'+day+'/modified/'+plate+'.csv'
                                try:
                                    bus_gps=pd.read_csv(agent_load_path,index_col=0,encoding='utf-8')
                                except:
                                    #print('fail to load',agent_load_path,type(plate))
                                    continue
                                location=get_location(bus['id'],bus_gps)
                        end_grid_id,end_time=get_destination(mon,day,location,sequence[-1]['record_id'],
                                                             sequence[-1]['key'],sequence[-1]['time'],final=True)
                        if end_grid_id!=-1:
                            sequence.append({'record_id':sequence[-1]['record_id'],'route':sequence[-1]['route'],
                                             'type':'bus_off','key':sequence[-1]['key'],'time':end_time,'grid_id':end_grid_id})
                    if sequence[-1]['type']!='bus_on':
                        sequences.append(sequence)
        processed+=1
        percent=round(processed/len(cards_matters)*100)
        show.add(percent)
    for sequence in sequences:
        for ac in sequence:
            ac['record_id']=int(ac['record_id'])
            
    agent_candidates={}
    for traj in sequences:
        source_id=str(traj[0]['grid_id'])
        des_id=str(traj[-1]['grid_id'])
        if source_id in agent_candidates:
            if des_id in agent_candidates[source_id]:
                agent_candidates[source_id][des_id]['count']+=1
                agent_candidates[source_id][des_id]['trajectories'].append(traj)
            else:
                agent_candidates[source_id][des_id]={'count':1,'trajectories':[traj]}
        else:
            agent_candidates[source_id]={des_id:{'count':1,'trajectories':[traj]}}
    try:
        trajectories=agent_candidates[str(source)][str(des)]['trajectories']
    except:
        trajectories=[]


    true_trajs=[]
    for traj in trajectories:
        for it in traj:
            it['record_id']=int(it['record_id'])
        if traj[-1]['time']!=0:
            true_trajs.append(traj)
    print('final number',len(true_trajs))
    return true_trajs




grid_info=json.load(open('/home/XXX/Python_Output/grids.json'))
grids=grid_info['grids']
station_index=['id','longitude','latitude','name']
subway_station_file='/home/XXX/167StationIDLocationsSorted.txt'
df_stations=pd.read_csv(subway_station_file,names=station_index)
lats=grid_info['latitude_list']
lngs=grid_info['longitude_list']
candidates_list=json.load(open('/home/XXX/Python_Output/003_IRL_Nonlinear/candidates.json'))

source_ids=[]
des_ids=[]
for candidate in candidates_list[0:300]:
    source_ids.append(candidate['source'])
    des_ids.append(candidate['destination'])
stations={}
for i in range(len(df_stations)):
    station=df_stations.iloc[i]
    name =station['name']
    if station['name'] in ['前海湾站','后海站','大剧院站','购物公园站','深圳北站']:
        stations[name]={'latitude':station['latitude'],'longitude':station['longitude']}
        name=name[:-1]
    elif station['name'] in ['前海湾','后海','大剧院','购物公园','深圳北']:
        stations[name]={'latitude':station['latitude'],'longitude':station['longitude']}
        name=name+'站'
    stations[name]={'latitude':station['latitude'],'longitude':station['longitude']}

import pyspark
confi=pyspark.SparkConf()
confi.set('spark.network.timeout','240s')
confi.set('spark.executor.memory','1500m')
confi.set('spark.driver.maxResultSize','1500m')
confi.set('spark.daemon.java.opts','-Xmx=10000m')
confi.set('spark.daemon.memory','10g')
confi.setMaster('local[20]')
sc = pyspark.SparkContext(conf=confi)




def map_function(i,source_id,des_id):
    mon='09'
    days=['05','06','09','26','28','30']
    if not os.path.exists('/home/XXX/Python_Output/trajectories/new_agent/'+str(i)+'/'):
        os.makedirs('/home/XXX/Python_Output/trajectories/new_agent/'+str(i)+'/')
    for day in days:
        trajectories=get_agent_m(mon,day,i,int(source_ids[i]),int(des_ids[i]))
        json.dump(trajectories,open('/home/XXX/Python_Output/trajectories/new_agent/'+str(i)+'/'+mon+day+'.json','w'))
        print('agent',i,'day',day,'finished')
    mon='11'
    days=['02','03','07','08','09','10','11']
    if not os.path.exists('/home/XXX/Python_Output/trajectories/new_agent/'+str(i)+'/'):
        os.makedirs('/home/XXX/Python_Output/trajectories/new_agent/'+str(i)+'/')
    for day in days:
        trajectories=get_agent_m(mon,day,i,int(source_ids[i]),int(des_ids[i]))
        json.dump(trajectories,open('/home/XXX/Python_Output/trajectories/new_agent/'+str(i)+'/'+mon+day+'.json','w'))
        print('agent',i,'day',day,'finished')
    return (i,'finished')

ids=list(range(len(source_ids)))

id_rdd=sc.parallelize(ids,150)
result=id_rdd.map(lambda d:map_function(d,int(source_ids[d]),int(des_ids[d]))).collect()
json.dump(result,open('/home/XXX/Python_Output/get_agent_result_statue.json','w'))

import hmac

def merge_card(agent):
    key = "XXXXXXXXXXXXXXXXX"
    mon=['09','11']
    days=[['05','06','09','26','28','30'],['02','03','07','08','09','10','11']]
    for i in range(len(mon)):
        for j in range(len(days[i])):
            mon_str = mon[i]
            day_str = days[i][j]
            try:
                data_json = json.load(open('/home/XXX/Python_Output/trajectories/new_agent/'+str(agent)+'/'+mon_str+day_str+'.json'))        
            except IOError:
                print('Error: Fail to open the file:/home/XXX/Python_Output/trajectories/new_agent/'+str(agent)+'/'+mon_str+day_str+'.json')           
            data_df = pd.concat(pd.DataFrame(d) for d in data_json)
            dict_1 = pd.read_csv('/home/XXX/Python_Output/card/'+mon_str+day_str+'.csv')
            dict_1=dict_1.drop_duplicates(['record_id','time'])
            merged = pd.merge(data_df,dict_1,on=['record_id','time'],how='left',left_index=True)
            merged['stage'] = data_df.index
            merged=merged.fillna(method='pad')
            merged['card'] = merged['card'].astype('int')
            merged.drop(['record_id'],axis=1,inplace = True)
            merged.drop(['Unnamed: 0'],axis=1,inplace = True)
            merged['key'] = [hmac.new(key,val).hexdigest() for val in merged['key']]
            merged['card'] = [hmac.new(key,val).hexdigest() for val in merged['card']]
            if not os.path.exists('/home/XXX/Python_Output/csvdata/'+str(agent)+'/'+mon_str+'/'):
                os.makedirs('/home/XXX/Python_Output/csvdata/'+str(agent)+'/'+mon_str+'/')
            merged.to_csv('/home/XXX/Python_Output/csvdata/'+str(agent)+'/'+mon_str+'/'+mon_str+day_str+'.csv',index=0)
            
for agent in range(0,300):
    merge_card(agent)
        