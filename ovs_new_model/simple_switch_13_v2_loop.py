# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from socket import timeout
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import tcp
from ryu.lib.packet import ether_types
import threading
from ryu.lib import hub

from operator import attrgetter
# from ryu.app 
import sys 
import os
# sys.path.append(os.path.abspath("/home/iot_team"))
# import simple_switch_13_v2_loop
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
import pandas as pd
import numpy as np
# from ryu.app import monitoring_loop as mn
import time
import os.path
from PyNomaly import loop


class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow = 0
        self.packet_in_cnt = 0
        self.datapaths = {}
        # self.monitor_thread = hub.spawn(self._monitor)
        self.pre_byte = 0
        self.cur_byte = 0
        self.pre_flow = 0
        self.cur_flow = 0
        # self.flow_cnt = 0
        self.flow_cnt_pre = 0
        self.data = np.zeros((1,2))
        self.data_arr = np.zeros((1,2))
        self.cycle = 1.0 # Collect data in each cycle

        #Collect training
        self.time_collect = 300.0 # time to collect training data in seconds
        self.time_cnt = 0.0
        self.write_done = False

        #Collect result
        self.lof_predict = np.ones((1,1))
        self.temp = np.ones((1,1))
        self.last_time =  int(time.time())
        self.cal = False
        self.last_cal = int(time.time())
        self.loop_training = 0

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install table-miss flow entry
        #
        # We specify NO BUFFER to max_len of the output action due to
        # OVS bug. At this moment, if we specify a lesser number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and
        # truncated packet data. In that case, we cannot output packets
        # correctly.  The bug has been fixed in OVS v2.1.0.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow_no_timeout(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        #print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            #print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=0)
        else:
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst, idle_timeout=0)
        #print(mod)
        # self.flow += 1
        datapath.send_msg(mod)

    def add_flow_no_timeout(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        #print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            #print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        #print(mod)
        # self.flow += 1
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        #print("state")
        datapath = ev.datapath
        #print(datapath)
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        #print("_monitor")
        # lof = LOF()
        # lof.train()
        # while True:
        # while not self.write_done:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            # Calculate the data
            #self.cal_new_flow()
            #print(self.flow_cnt)
            self.time_cnt += self.cycle
            print(self.time_cnt)
            self.data[0,1] = (self.packet_in_cnt - self.flow_cnt_pre)/self.cycle
            self.flow_cnt_pre = self.packet_in_cnt
            self.data_arr = np.concatenate((self.data_arr,self.data), axis=0)
            # self.temp[0,0] = lof.model.predict(lof.normalize_point(self.data))[0]     #
            # self.lof_predict = np.concatenate((self.lof_predict, self.temp))
            #print(lof.model.predict(lof.normalize_point(self.data)))
            if(self.time_cnt > self.time_collect and self.write_done== False):
                # self.train_to_csv()
                # self.to_csv()
                self.data_arr = np.delete(self.data_arr, 0, 0)
                self.data_arr = np.delete(self.data_arr, 0, 0)
                np.savetxt("/home/iot_team/DATN/training.csv", self.data_arr, delimiter=',')
                self.write_done = True
                print(">>>>>>>>>>>> Write done!")

            #print(self.data)
            print("------------------------------------------------------------")
            # hub.sleep(self.cycle)

    def _request_stats(self, datapath):
        #print("_request_stats")
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        #req = parser.OFPFlowStatsRequest(datapath)
        #datapath.send_msg(req)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    '''def cal_new_flow(self):
        #print("cal_new_flow")
        self.data[0,1] = (self.cur_flow - self.pre_flow)/self.cycle
        self.pre_flow = self.cur_flow
        self.cur_flow = 0'''


    '''
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        #print("_flow_stats_reply_handler")
        body = ev.msg.body
        #dp = ev.msg.datapath
        #dpid = dp.id
        for stat in body:
            if('in_port' in stat.match and stat.match['in_port'] == 1):
                self.cur_flow += 1
                #print(self.cur_flow)'''

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        #print("_port_stats_reply_handler")
        body = ev.msg.body
        dp = ev.msg.datapath
        ofproto = dp.ofproto
        dpid = dp.id
        port_no = 1
        msg = ev.msg
        
        for stat in body:
            if(stat.port_no == port_no):
                print("Im working!!!")
                #port_no = stat.port_no
                self.cur_byte = stat.rx_bytes
                #print(self.cur_byte)
                #print(self.pre_byte)
                throughput = (self.cur_byte - self.pre_byte)*8.0/self.cycle
                #print(throughput)
                #print("--------------------------------")
                self.data[0,0] = throughput
                self.pre_byte = stat.rx_bytes

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        self.packet_in_cnt += 1
        #print(self.count)
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if(len(pkt.get_protocols(ipv4.ipv4)) != 0):
            ip = pkt.get_protocols(ipv4.ipv4)[0] ###################### Myself
            ip_src = ip.src ####################### Myself
            ip_dst = ip.dst ####################### Myself
            #print(ip)
            ip_proto = ip.proto
            if(len(pkt.get_protocols(tcp.tcp)) != 0):
                tcp_mod = pkt.get_protocols(tcp.tcp)[0]
                tcp_src = tcp_mod.src_port
                tcp_dst = tcp_mod.dst_port
                #print(tcp_mod)
                #ip_proto = ip.proto
        #print(pkt)
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            #print("AAAAAAAAAAAAAAAAAAA")
            return
        eth_type = eth.ethertype
        dst = eth.dst
        src = eth.src
        #ip_src = ip.src ####################### Myself
        #ip_dst = ip.dst ####################### Myself

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        #print("BBBBBBBBBBBBBBBBBBBBBBBB")
        #self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
            #print("CCCCCCCCCCCCCCCCCCCCCCC")
        else:
            out_port = ofproto.OFPP_FLOOD
            #print("DDDDDDDDDDDDDDDDDDDDDDD")

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            #print("EEEEEEEEEEEEEEEEEEEEEE")
            if (len(pkt.get_protocols(ipv4.ipv4)) != 0):
                #print("FFFFFFFFFFFFFFFFFFFFFFFFF")
                if( len(pkt.get_protocols(tcp.tcp))  == 0):
                    match = parser.OFPMatch(in_port=in_port,
                                            eth_dst=dst,
                                            eth_src=src,
                                            eth_type=eth_type,
                                            ipv4_src=ip_src,
                                            ipv4_dst=ip_dst,
                                            ip_proto=ip_proto)
                else: 
                    match = parser.OFPMatch(in_port=in_port,
                                            eth_dst=dst,
                                            eth_src=src,
                                            eth_type=eth_type,
                                            ipv4_src=ip_src,
                                            ipv4_dst=ip_dst,
                                            ip_proto=ip_proto,
                                            tcp_src=tcp_src,
                                            tcp_dst=tcp_dst)
                # verify if we have a valid buffer_id, if yes avoid to send both
                # flow_mod & packet_out
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    #print("GGGGGGGGGGGGGGGGGGGGG")
                    self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                    #return
                else:
                    #print("HHHHHHHHHHHHHHHHHHHHHH")
                    self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            #print("IIIIIIIIIIIIIIIIIIIIIIIIIIII")
            data = msg.data
        #print("KKKKKKKKKKKKKKKKKKKKKKKKKKK")
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        #print("LLLLLLLLLLLLLLLLLLLLLLLLLLLL")
        #print(out)
        #print("\n")
        datapath.send_msg(out)
        # t1 = threading.Thread(target=mn.SimpleMonitor13._monitor(self), name='t1')
        # t1.start()

        now = int(time.time())
        if (now-self.last_time) >= 1:
            print('hello')
            if os.path.exists("/home/iot_team/DATN/training.csv"):
                self.last_time = now + 10000
                self.cal = True
                self.loop_training = loop.LocalOutlierProbability(self.data_arr, extent=3, n_neighbors=20).fit()
                scores = self.loop_training.local_outlier_probabilities
                print(len(scores))
                print(scores)
                print('loop cuoi =  {}'.format(scores[-1]))
                array = np.array(self.data_arr[-1][-2], self.data_arr[-1][-1])
                print(self.loop_training.stream(array))
                # print(self.loop_training.stream(self.data_arr[48]))
                
            else:
                self.last_time = now
            self._monitor()

        now1 = int(time.time())
        if (now1-self.last_cal) >= 1:
            self.last_cal = now1
            if self.cal:
                for dp in self.datapaths.values():
                  self._request_stats(dp)
                # Calculate the data
                #self.cal_new_flow()
                #print(self.flow_cnt)
                self.time_cnt += self.cycle
                print(self.time_cnt)
                self.data[0,1] = (self.packet_in_cnt - self.flow_cnt_pre)/self.cycle
                self.flow_cnt_pre = self.packet_in_cnt
                # array = np.array(self.data[0][0], sel)
                print("traffic = {} - flow = {}".format(self.data[0][0], self.data[0][1]))
                array = np.array([self.data[0][0], self.data[0][1]])
                print(self.loop_training.stream(array))

    
        

        # hub.spawn(mn.SimpleMonitor13._monitor(self))

    
# class SimpleMonitor13(SimpleSwitch13):
  
#     def __init__(self, *args, **kwargs):
#         super(SimpleMonitor13, self).__init__(*args, **kwargs)
#         #print("init")
#         self.datapaths = {}
#         self.monitor_thread = hub.spawn(self._monitor)
#         self.pre_byte = 0
#         self.cur_byte = 0
#         self.pre_flow = 0
#         self.cur_flow = 0
#         # self.flow_cnt = 0
#         self.flow_cnt_pre = 0
#         self.data = np.zeros((1,2))
#         self.data_arr = np.zeros((1,2))
#         self.cycle = 1.0 # Collect data in each cycle

#         #Collect training
#         self.time_collect = 300.0 # time to collect training data in seconds
#         self.time_cnt = 0.0
#         self.write_done = False

#         #Collect result
#         self.lof_predict = np.ones((1,1))
#         self.temp = np.ones((1,1))

#     @set_ev_cls(ofp_event.EventOFPStateChange,
#                 [MAIN_DISPATCHER, DEAD_DISPATCHER])
#     def _state_change_handler(self, ev):
#         #print("state")
#         datapath = ev.datapath
#         #print(datapath)
#         if ev.state == MAIN_DISPATCHER:
#             if datapath.id not in self.datapaths:
#                 self.logger.debug('register datapath: %016x', datapath.id)
#                 self.datapaths[datapath.id] = datapath
#         elif ev.state == DEAD_DISPATCHER:
#             if datapath.id in self.datapaths:
#                 self.logger.debug('unregister datapath: %016x', datapath.id)
#                 del self.datapaths[datapath.id]

#     def _lop(self):
#         while True:
#             print('Tuan')
#             hub.sleep(1)

#     def _monitor(self):
#         #print("_monitor")
#         # lof = LOF()
#         # lof.train()
#         # while True:
#         while not self.write_done:
#             for dp in self.datapaths.values():
#                 self._request_stats(dp)
#             # Calculate the data
#             #self.cal_new_flow()
#             #print(self.flow_cnt)
#             self.time_cnt += self.cycle
#             self.data[0,1] = (self.packet_in_cnt - self.flow_cnt_pre)/self.cycle
#             self.flow_cnt_pre = self.packet_in_cnt
#             self.data_arr = np.concatenate((self.data_arr,self.data), axis=0)
#             # self.temp[0,0] = lof.model.predict(lof.normalize_point(self.data))[0]     #
#             # self.lof_predict = np.concatenate((self.lof_predict, self.temp))
#             #print(lof.model.predict(lof.normalize_point(self.data)))
#             if(self.time_cnt > self.time_collect and self.write_done== False):
#                 # self.train_to_csv()
#                 # self.to_csv()
#                 np.savetxt("/home/iot_team/DATN/training.csv", self.data_arr, delimiter=',')
#                 self.write_done = True
#                 print(">>>>>>>>>>>> Write done!")

#             #print(self.data)
#             print("------------------------------------------------------------")
#             hub.sleep(self.cycle)

#     def to_csv(self):
#         np.savetxt("/home/clientserver/Desktop/lof_ryu_nghia/test_cases/test_data/test.csv", self.data_arr, delimiter=',')
#         # np.savetxt("./test_cases/k=6/label_lab_3_edit_00013.csv", self.lof_predict, delimiter=',')

#     def train_to_csv(self):
#         '''Use to collect training data to csv file '''
#         np.savetxt("/home/clientserver/Desktop/lof_ryu_nghia/test_cases/train_dataset_test.csv", self.data_arr, delimiter=',')

#     def _request_stats(self, datapath):
#         #print("_request_stats")
#         ofproto = datapath.ofproto
#         parser = datapath.ofproto_parser
#         dpid = datapath.id
#         #req = parser.OFPFlowStatsRequest(datapath)
#         #datapath.send_msg(req)
#         req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
#         datapath.send_msg(req)

#     '''def cal_new_flow(self):
#         #print("cal_new_flow")
#         self.data[0,1] = (self.cur_flow - self.pre_flow)/self.cycle
#         self.pre_flow = self.cur_flow
#         self.cur_flow = 0'''


#     '''
#     @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
#     def _flow_stats_reply_handler(self, ev):
#         #print("_flow_stats_reply_handler")
#         body = ev.msg.body
#         #dp = ev.msg.datapath
#         #dpid = dp.id
#         for stat in body:
#             if('in_port' in stat.match and stat.match['in_port'] == 1):
#                 self.cur_flow += 1
#                 #print(self.cur_flow)'''

#     @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
#     def _port_stats_reply_handler(self, ev):
#         #print("_port_stats_reply_handler")
#         body = ev.msg.body
#         dp = ev.msg.datapath
#         ofproto = dp.ofproto
#         dpid = dp.id
#         port_no = 1
#         msg = ev.msg
        
#         for stat in body:
#             if(stat.port_no == port_no):
#                 print("Im working!!!")
#                 #port_no = stat.port_no
#                 self.cur_byte = stat.rx_bytes
#                 #print(self.cur_byte)
#                 #print(self.pre_byte)
#                 throughput = (self.cur_byte - self.pre_byte)*8.0/self.cycle
#                 #print(throughput)
#                 #print("--------------------------------")
#                 self.data[0,0] = throughput
#                 self.pre_byte = stat.rx_bytes

