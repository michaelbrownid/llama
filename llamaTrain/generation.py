# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        #MPSB torch.set_default_tensor_type(torch.cuda.HalfTensor)
        torch.set_default_tensor_type(torch.FloatTensor) # on CPU not GPU!
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode(False)
    def backprop(
        self,
        prompts: List[str],
    ):

        """For now assume that all prompts are the same length for
        simplicity not having to track the end of sentence

        tokens will simply have have the BOS=1 at the begining and
        will NOT have the last token to be predicted

        prompt_tokens = [1,    1763, 367, 470, 451]
                        "BOS   to    be   or   not"

        I want the input to be:
               tokens = [1,    1763, 367, 470]
                        "BOS   to    be   or"

        I want the correct answer to be:
              correct = [1763, 367,  470, 451]
                        "to    be    or   not"
        
        """

        targetDeviceMPSB = "cpu"

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=targetDeviceMPSB)

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len


        #### learning
        learning_rate = 1.0E-03
        optimizer = torch.optim.SGD(self.model.parameters(),lr=learning_rate)
        lossfunc = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        
        #### Now compute the model
        prev_pos = 0
        logits = self.model.forward(tokens[:, :-1], prev_pos) # forward on everything but last in tokens [1,4,32000]

        #### loss at every position
        target = tokens[:, 1:] # target is the shifted second to last in tokens [1,4]
        loss = lossfunc(logits.transpose(1,2),target) # transpose as this is as CrossEntropyLoss wants it
        print("loss",loss.item() )

        # #### loss only at LAST token
        # target = tokens[:, -1] # target is the last token
        # loss = lossfunc(logits[:,-1],target) # only the predictions at the last
        # print("loss",loss.item() )
        
        if True:
            print ("logprobs ------")
            myprobs = F.softmax(logits, dim=-1)
            for bb in range(target.shape[0]):
                for tt in range(target.shape[1]):
                    print("bb,tt,logits[target]",bb,tt,
                          logits[bb,tt,target[bb,tt]].item(),
                          myprobs[bb,tt,target[bb,tt]].item() )
                    
        loss.backward()

        optimizer.step()

        print("optimizer took step")


