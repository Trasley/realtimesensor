#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import statistics
import threading
from time import sleep
import numpy as np
from math import sqrt
import sys
import json
import argparse
import pandas as pd
import pickle
import logging
import re

class TCPClient(threading.Thread):

    def __init__(self, host, port,modelpkl):
        super().__init__()
        self.host = host
        self.port = port
        self.occupancy = {"Occupancy": None}
        self.model = pickle.load(open(modelpkl, 'rb'))
        self.logger = logging.getLogger('Client')
        self.regex = re.compile('points')

    def work_with_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:

            # Connect via TCP
            sock.connect((self.host, self.port))

            while True:
                # receive from server
                received = str(sock.recv(1024), "utf-8").rstrip('\t\n')
                if not received:
                    #No data sent - exit
                    break
                elif self.regex.search(received):
                    #final points total sent
                    print('Received ' + received)
                    break
                else:
                    # sensor data sent
                    print('Received ' + received)
                    self.logger.info('Received: ' + received)
                    dataload = json.loads(received)
                    df = pd.DataFrame(dataload, index=['Number'])

                    #Pass data to model & predict
                    room_state = df.iloc[:, 2:7]
                    prediction = self.model.predict(room_state).tolist()[0]
                    self.occupancy["Occupancy"] = prediction

                    #Format response & send
                    occupancy_string = json.dumps(self.occupancy).encode('utf-8')
                    sock.send(occupancy_string)
                    print('Sent ' + str(self.occupancy.items()))
                    self.logger.info('Sent: ' + str(self.occupancy.items()))


        except Exception as issue:
            print(issue)
        finally:
            # clean up
            sock.close()


    def run(self):
        self.work_with_server()


logging.basicConfig(level=logging.DEBUG, filename='client_log.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger()
TCPClient('localhost', 9999,'logistic_model.sav').work_with_server()
