import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations.
        
        Interface:
            Observation:
                - observation/wrist_image_left: (H, W, 3) if needs_wrist_camera is True
                - observation/exterior_image_{i}_left: (H, W, 3) if n_external_cameras >= 1
                - observation/exterior_image_{i}_right: (H, W, 3) if needs_stereo_camera is True
                - session_id: (1,) if needs_session_id is True
                - observation/joint_position: (7,)
                - observation/gripper_position: (1,)
                - prompt: str, the natural language task instruction for the policy
            
            Action:
                - action: (N, 8,), 7d joint action in specified action space + gripper position
                                --> all N actions will get executed on the robot before the server is queried again
        """

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass