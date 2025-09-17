import numpy as np
import logging

import roboarena.policy_server as policy_server
from roboarena.policy_client import WebsocketClientPolicy
from roboarena.policy import BasePolicy


def _make_dummy_observation(server_config: policy_server.PolicyServerConfig) -> dict:
    obs = {}
    if server_config.image_resolution is not None:
        h, w = server_config.image_resolution
    else:
        h, w = 288, 512
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    if server_config.n_external_cameras > 0:
        for i in range(server_config.n_external_cameras):
            obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
            if server_config.needs_stereo_camera:
                obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    if server_config.needs_session_id:
        obs["session_id"] = "38DSA3hd3"
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = "test"
    return obs


def test_policy_server():
    client = WebsocketClientPolicy()

    # Make sure metadata is reasonable
    metadata = client.get_server_metadata()
    assert isinstance(metadata, dict)
    try:
        server_config = policy_server.PolicyServerConfig(**metadata)
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        raise e
    
    assert server_config.n_external_cameras in [0, 1, 2]
    assert server_config.action_space in ["joint_position", "joint_velocity"]

    # Make sure we can get a response
    obs = _make_dummy_observation(server_config)
    actions = client.infer(obs)
    assert isinstance(actions, np.ndarray)
    assert len(actions.shape) == 2
    assert actions.shape[-1] == 8

    # At the end of the trajectory, reset will be called
    client.reset()

    logging.info("Test passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_policy_server()
