"""Example client for communicating with a policy server.

Code copied from https://github.com/Physical-Intelligence/openpi.

"""

import logging
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from roboarena.policy import BasePolicy
from roboarena.utils import msgpack_numpy


class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except:
            logging.info("Connection to server with ws:// failed. Trying wss:// ...")
            
        self._uri = "wss://" + self._uri.split("//")[1]
        conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # Notify server that we're calling the infer endpoint (as opposed to the reset endpoint)
        obs["endpoint"] = "infer"

        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, reset_info: Dict) -> None:
        # Notify server that we're calling the reset endpoint (as opposed to the infer endpoint)
        reset_info["endpoint"] = "reset"

        data = self._packer.pack(reset_info)
        self._ws.send(data)
        response = self._ws.recv()
        return response

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy()
    actions_dict = client.infer({})
    actions = actions_dict["actions"]
    print(f"Actions received: {actions}")
    client.reset({})
