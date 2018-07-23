"""
Python2 script
"""

import argparse
import logging

from multiprocessing.connection import Listener

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def listen(args):
    address = (args.server_ip, args.server_port)
    listener = Listener(address)

    try:
        # Listen for connections
        while True:
            logger.info("Awaiting connection")
            socket = listener.accept()
            logger.info("Accepted connection from {}".format(listener.last_accepted))

            try:
                # Listen for commands
                while True:
                    ctrl, resource = socket.recv()
                    if ctrl == 'caption':
                        socket.send("something")

            except EOFError as e:
                logger.warning("Client left unexpectedly {}".format(e.message))
            except Exception as e:
                logger.exception(e)
            finally:
                socket.close()
    except Exception as e:
        logger.exception(e)
    finally:
        listener.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', help="IP to listen on for captioning requests", default="localhost")
    parser.add_argument('--server_port', type=int, help="Port to listen on for captioning requests.", default=45999)

    args = parser.parse_args()

    listen(args)
