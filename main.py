#!/usr/local/bin/python

import numpy as np
import random
import time
import os
import tensorflow as tf

from sqlalchemy import Column,String,create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base 

from utils.cfg import load_cfg
from utils.data import  get_local_data, get_real_data, get_data

LEARNING_RATE_BASE = 0.0081
REGULARIZATION_RATE = 0.0001

#Basic Network Architecture
class NN(object): 
    def __init__(self,Nodes,lr,norm=True):
        self.INPUTNODES=Nodes[0]
        self.INPUTSHAPE=[None,self.INPUTNODES]
        self.HIDDENNODES=Nodes[1:-1]
        self.OUTPUTNODES=Nodes[-1]
        self.OUTPUTSHAPE=[None,self.OUTPUTNODES]
        self.LABELSHAPE=[None]

        self.sess=tf.InteractiveSession()
        self.x=tf.placeholder(tf.float32,self.INPUTSHAPE,name="x")
        self.y_=tf.placeholder(tf.int32,self.LABELSHAPE,name="y_")
        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.Variable(initial_value=lr,trainable=False)
        self.is_bn = norm
        self.rate = tf.placeholder(tf.float32)

        self.inference()
        global_step=tf.Variable(0,False)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_,logits=self.y)
        with tf.name_scope('cross_entropy'):
            self.loss=tf.reduce_mean(cross_entropy)+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            with tf.name_scope('total'):
                cross_entropy=tf.reduce_mean(self.loss)
        tf.summary.scalar('cross_entropy',cross_entropy)
        self.train_writer = tf.summary.FileWriter("./train_process", self.sess.graph)
        self.merged = tf.summary.merge_all()

        self.update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.lr=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,50,0.96)
            self.train_op=tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=global_step)
        self.correct_prediction=tf.equal(tf.cast(tf.argmax(self.y,1),tf.int32),tf.cast(self.y_,tf.int32))
        self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        
        self.sess=tf.Session()
        self.saver=tf.train.Saver()

    def add_layer(self,layer_name,inputdata,insize,outsize,activate=True,norm=True,drp=True):
        with tf.variable_scope(layer_name):
            weights=tf.get_variable("weights",shape=[insize,outsize],\
                initializer=tf.truncated_normal_initializer(stddev=0.5),\
                    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE))
            print(weights)
            biases=tf.get_variable("biases",shape=[outsize],initializer=tf.constant_initializer(0.0),\
                regularizer=None)
            z=tf.matmul(inputdata,weights)+biases

            if norm:
                bn=tf.contrib.layers.batch_norm(z,decay=0.99,epsilon=1e-5,center=True,scale=True,is_training=self.is_train,scope="bn")
                print(layer_name+'_bn')
            else:
                bn=z
            if activate:
                a=tf.nn.relu(bn)
            else:
                a=bn
            if drp:
                print(layer_name+'_dropout')
                return tf.nn.dropout(a,self.rate)
            else:
                return a
    
    def inference(self):
        self.h=self.add_layer("hidden_layer1",self.x,self.INPUTNODES,self.HIDDENNODES[0],norm=self.is_bn)
        for i in range(1,len(self.HIDDENNODES)):
            layer_name='hidden_layer'+str(i+1)
            self.h=self.add_layer(layer_name,self.h,self.HIDDENNODES[i-1],self.HIDDENNODES[i],norm=self.is_bn)
        self.y=self.add_layer("output_layer",self.h,self.HIDDENNODES[-1],self.OUTPUTNODES,activate=False,norm=False,drp=False)
        return
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
    
    def train(self,x,y):
        self.summary , _=self.sess.run([self.merged,self.train_op],feed_dict={self.x:x,self.y_:y,self.is_train:True,self.rate:1})

    def compute_loss(self,x,y):
        return self.sess.run(self.loss,feed_dict={self.x:x,self.y_:y,self.is_train:False,self.rate:1})

    def compute_accuracy(self,x,y):
        return self.sess.run(self.accuracy,feed_dict={self.x:x,self.y_:y,self.is_train:False,self.rate:1})

    def predict(self,x,y):
        return self.sess.run(tf.argmax(tf.nn.softmax(self.y),1),feed_dict={self.x:x,self.y_:y,self.is_train:False,self.rate:1})

    def read_model(self,path):
        self.saver.restore(self.sess,path+"/bp.ckpt")

    def save_model(self,path):
        self.saver.save(self.sess,path+"/bp.ckpt")

    def changelr(self,lr):
        self.sess.run(tf.assign(self.lr,lr))    

#Network Apply 
class PN(object):
    def __init__(self,_input,_output,lr,norm=False):
        nodes=[_input,30,20,15,10,_output]
        self.nn=NN(nodes,lr,norm)
    
    def test(self,Home_Data,Virtual_Label):
        pre=self.nn.predict(Home_Data,Virtual_Label)
        return pre

def  statistic(Pre_Label,Device_List):
    All_Device={}.fromkeys(Device_List).keys()
    All_Fluorine=[]
    for temp in All_Device:
        Single_Pre=[]
        for i in range(len(Device_List)):
            if Device_List[i]==temp:
                Single_Pre.append(Pre_Label[i])
        Every_Pre={}.fromkeys(Single_Pre).keys()
        Pre_Pro=[]
        for j in Every_Pre:
            num=Single_Pre.count(j)
            proportion=num/len(Single_Pre)
            Pre_Pro.append(proportion)
        temp_dict=dict(zip(Every_Pre,Pre_Pro))
        fl=int(max(temp_dict.items(), key=lambda x: x[1])[0])
        fl=(fl+2)*0.1
        fl=format(fl, '.0%')
        All_Fluorine.append(fl)
    return All_Device,All_Fluorine

def get_running_data(maclist,name,begin_time,end_time):
    beginTime = begin_time
    endTime = end_time
    
    raw_data=[]
    for i in range(len(maclist)):
        temp = get_local_data([maclist[i]], beginTime, endTime, DeviceFlag = 'dehumidifier')
        raw_data = get_data(temp, raw_data, name[i])

    raw_data = np.array(raw_data)
    Home_Data, Virtual_List, Device_List = get_real_data(raw_data)
    return Home_Data, Virtual_List, Device_List

def main():
    cfg = load_cfg("./configs/dehumidifier.yaml")

    pn = PN(30,7,0.0003,norm=False)
    pn.nn.read_model(cfg['model']['path'])
    maclist = cfg['device']['mac']
    name = cfg['name']['fluorine']
    begin_time = cfg['time']['begin']
    end_time = cfg['time']['end']

    Home_Data, Virtual_Label, Device_List = get_running_data(maclist, name, begin_time, end_time)
    Pre_Label = pn.test(Home_Data, Virtual_Label)
    All_Device, All_Fluorine = statistic(Pre_Label, Device_List)

    Base = declarative_base()
    class Dehumidifier(Base):
        __tablename__ = 'record_fluorine_dehumidifier'
        mac = Column(String(20), primary_key=True)
        fluorine = Column(String(20))

        def __init__(self, mac, fluorine):
            self.mac = mac
            self.fluorine = fluorine
        
        def replace(self, session):
            existing = session.query(Dehumidifier).filter_by(mac=self.mac).one_or_none()
            if not existing:
                session.add(self)
            else:
                existing.fluorine = self.fluorine
            session.commit()

    engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(
        cfg['mysql']['user'],
        cfg['mysql']['passwd'],
        cfg['mysql']['host'],
        cfg['mysql']['port'],
        cfg['mysql']['db']))

    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    All_Device = list(All_Device)
    All_Fluorine = list(All_Fluorine)
    for i in range(len(All_Device)):
        mac = All_Device[i]
        fluorine = All_Fluorine[i]
        new_dehumidifier = Dehumidifier(mac, fluorine)
        new_dehumidifier.replace(session)
    session.close()

if __name__ == "__main__":
    main() 