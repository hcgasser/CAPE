#!/usr/bin/env python

import socket
import threading
import pickle
import json
import argparse

from kit.log import setup_logger, log_info
from kit.bioinf.mhc.PWM import PWMPredictor

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--port', type=int, default=12345, help="the port for the server")
argparser.add_argument('--pwm', type=str, help="folder with the position weight matrices", required=True)
args = argparser.parse_args() 
setup_logger()

# Define the host and port to listen on
HOST = 'localhost'
PORT = args.port

predictor = None


# this function performs the actual MHC prediction request
def perform_action(obj, pwm_folder):
    global predictor, cnt
    peptide = str(obj["peptide"]).strip()
    alleles = str(obj["alleles"]).strip()
    alleles = eval(alleles)

    # if cnt == 0:
    #     print(alleles)
    # cnt += 1
    # return 0

    if predictor is None:
        predictor = PWMPredictor(folder=pwm_folder)

    result = 0
    for allele in alleles:
        if allele not in predictor.PWMs_log:
            predictor.load_allele(allele, 9)

        cnt += 1
        if cnt % 1e6 == 0:
            print(f"Peptide-MHC evaluations: {cnt:>10,}", end="\r")

        presented = predictor.peptide_presented(peptide, allele)
        if presented or presented is None:
            result += 1
        # res = predictor.peptide_rank(peptide, allele, predict_if_missing=True)
        # if res < 0.02:
        #    result += 1

    return result


# this function handles a single client connection
def handle_connection(conn, addr, pwm_folder):
    global cnt
    log_info(f'Connected by {addr}')
    cnt = 0

    while True:
        # Receive data from the client
        data = conn.recv(1024)

        if not data:
            # If the client closes the connection, break out of the loop
            break

        data = data.decode('utf-8')
        # Parse the JSON string sent by the client
        obj = json.loads(data)

        # Perform an action based on the object and get the result
        result = perform_action(obj, pwm_folder)

        # Pickle the result and send it back to the client
        conn.sendall(bytes(str(result), "UTF-8"))

    # Close the connection
    conn.close()
    log_info(f'Connection closed by {addr} after {cnt} Peptide-MHC evaluations')


# Create a socket object that is able to handle several connections at the same time
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Bind the socket to the specified host and port

    log_info(f"Start Server: {HOST}:{PORT}")
    s.bind((HOST, PORT))

    while True:
        # Listen for incoming connections
        s.listen()

        # Wait for a client to connect
        conn, addr = s.accept()

        handle_connection(conn, addr, args.pwm)
        # threading.Thread(target=handle_connection, args=(conn, addr)).start()
