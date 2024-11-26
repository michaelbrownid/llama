# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        #"I am going to Paris, what should I see? In Haiku form the answer is",
        #"To be or not",
        "To be, or not to be, that is the question:",
        #"",
        
    ]

    ### do a few rounds of backprop
    results = generator.backprop(prompts,)
    #results = generator.backprop(prompts,)
    #results = generator.backprop(prompts,)

    #### save the new model
    torch.save(generator.model.state_dict(), "trainedModel.pth")

if __name__ == "__main__":
    fire.Fire(main)
