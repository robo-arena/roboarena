import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations.
        
        Interface:
            Observation:
                - observation/wrist_image_left: (H, W, 3) if needs_wrist_camera is True
                - observation/wrist_image_right: (H, W, 3) if needs_wrist_camera is True and needs_stereo_camera is True
                - observation/exterior_image_{i}_left: (H, W, 3) if n_external_cameras >= 1
                - observation/exterior_image_{i}_right: (H, W, 3) if needs_stereo_camera is True
                - session_id: (1,) if needs_session_id is True
                - observation/joint_position: (7,)
                - observation/cartesian_position: (6,)
                - observation/gripper_position: (1,)
                - prompt: str, the natural language task instruction for the policy
            
            Action:
                - action: (N, 8,) or (N, 7,): either 7 movement actions (for joint action spaces) or 6 (for cartesian) plus one dimension for gripper position
                                --> all N actions will get executed on the robot before the server is queried again
        """

    def reset(self, reset_info: Dict) -> None:
        """In case that the policy is stateful, delete the tracked state for the policy rollout uniquely defined by reset_info["session_id"]"""
        pass
