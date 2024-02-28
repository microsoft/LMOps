from dataclasses import dataclass

@dataclass
class ResLoraConfig:
    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    start_index : int = -100
    target_modules: str = "q.k.v.o.f"
    lora_num: int = 1
    merge_weights: bool = True

    res_flag: int = 0
    merge_flag: int = 0
    pre_num: int = 0
    merge_4_len: int = 100

    
    def to_json_string(self):
        return self.__dict__
