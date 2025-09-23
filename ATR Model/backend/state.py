from dataclasses import dataclass


@dataclass
class TrainingState:
    last_uploaded: str = ""
    is_training: bool = False
    trained: bool = False
    progress: int = 0
    artifact: str = ""
    current_stage: str = ""


