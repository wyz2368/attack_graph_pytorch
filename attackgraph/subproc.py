""" Subprocessing utility functions, used in Gambit. """
import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)


def call_and_wait_with_timeout(command_str, timeout):
    logger.info("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True, preexec_fn=os.setsid)
    timeout_seconds = timeout
    try:
        my_process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.critical("Process ran more seconds than: " + str(timeout_seconds))
        os.killpg(os.getpgid(my_process.pid), signal.SIGTERM)
        logger.critical("Subprocess has been killed.")
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()


def call_and_wait(command_str):
    logger.info("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True)
    my_process.wait()
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()


def call_and_wait_with_timeout_and_check(command_str):
    logger.info("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True)
    timeout_seconds = 3600
    try:
        my_process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning("Process ran more seconds than: " + str(timeout_seconds))
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()
