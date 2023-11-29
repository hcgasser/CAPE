#!/usr/bin/env python

import socket
import json
import argparse

from kit.log import setup_logger, log_info
from kit.bioinf.immuno.mhc_1._pwm import Mhc1PredictorPwm
from kit.bioinf.immuno.mhc_1 import MHC_1_PEPTIDE_LENGTHS

from CAPE.RL.reward import get_visible_natural

MHC_1_PREDICTOR = None


# this function performs the actual MHC prediction request
def perform_action(obj, pwm_path, input_files_path, cnt):
    global MHC_1_PREDICTOR
    peptide = str(obj["peptide"]).strip()
    alleles = str(obj["alleles"]).strip()
    alleles = eval(alleles)

    rewards = str(obj["rewards"]).strip()
    res = rewards.split(",")
    reward_visible_artificial, reward_visible_natural = float(res[0]), float(res[1])

    if reward_visible_natural != reward_visible_artificial:
        visible_natural = get_visible_natural(
            MHC_1_PREDICTOR, alleles, MHC_1_PEPTIDE_LENGTHS, input_files_path
        )
    else:
        visible_natural = set()

    if MHC_1_PREDICTOR is None:
        MHC_1_PREDICTOR = Mhc1PredictorPwm(data_dir_path=pwm_path)

    result = 0
    for allele in alleles:
        if allele not in MHC_1_PREDICTOR.PWMs_log:
            MHC_1_PREDICTOR.load_allele(allele, 9)

        presented = MHC_1_PREDICTOR.peptide_presented(peptide, allele)
        if presented or presented is None:
            if presented in visible_natural:
                result += reward_visible_natural
            else:
                result += reward_visible_artificial

        cnt += 1
        if cnt % 1e6 == 0:
            print(f"Peptide-MHC evaluations: {cnt:>10,}", end="\r")

    return result, cnt


# this function handles a single client connection
def handle_connection(conn, addr, pwm_path, input_files_path):
    log_info(f"Connected by {addr}")

    cnt = 0
    while True:
        # Receive data from the client
        data = conn.recv(1024)

        if not data:
            # If the client closes the connection, break out of the loop
            break

        data = data.decode("utf-8")
        # Parse the JSON string sent by the client
        obj = json.loads(data)

        # Perform an action based on the object and get the result
        result, cnt = perform_action(obj, pwm_path, input_files_path, cnt)

        # Pickle the result and send it back to the client
        conn.sendall(bytes(str(result), "UTF-8"))

    # Close the connection
    conn.close()
    log_info(f"Connection closed by {addr} after {cnt} Peptide-MHC evaluations")


def main(_args):
    setup_logger()

    # Define the host and port to listen on
    host = "localhost"
    port = _args.port

    # Create a socket object that is able to handle several connections at the same time
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to the specified host and port

        log_info(f"Start Server: {host}:{port}")
        s.bind((host, port))

        while True:
            # Listen for incoming connections
            s.listen()

            # Wait for a client to connect
            conn, addr = s.accept()

            handle_connection(conn, addr, _args.pwm_path, _args.input_files_path)
            # threading.Thread(target=handle_connection, args=(conn, addr)).start()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--port", type=int, default=12345, help="the port for the server"
    )
    argparser.add_argument(
        "--pwm_path",
        type=str,
        help="folder with the position weight matrices",
        required=True,
    )
    argparser.add_argument(
        "--input_files_path",
        type=str,
        help="input files path",
        default="",
        required=False,
    )
    args = argparser.parse_args()

    main(args)
