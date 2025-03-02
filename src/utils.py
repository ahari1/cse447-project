from collections import defaultdict
import numpy as np
import os
import subprocess
import sys
import marisa_trie
import time
from concurrent.futures import ThreadPoolExecutor

def get_llama():
    try:
        subprocess.check_output('nvidia-smi')
        print('Nvidia GPU detected!', flush=True)
        has_cuda = True
        sys.path.insert(0, "/opt/llama-cpp-cuda")
    except Exception:
        print("No Nvidia GPU detected!", flush=True)
        has_cuda = False
    from llama_cpp import Llama
    return Llama, has_cuda

def load_bloom(work_dir="../work"):
    # Load model and tokenizer
    # https://huggingface.co/docs/transformers/en/gguf
    Llama, use_gpu = get_llama()
    filename = "bloom-560m.q8_0.gguf"
    model = Llama(model_path=os.path.join(work_dir, filename), logits_all=True, n_gpu_layers=-1 if use_gpu else 0, verbose=False)

    # load token vocabulary
    NUM_TOKENS = model.n_vocab()
    token_vocab = []
    for token in range(NUM_TOKENS):
        try:
            decoded_token = model.detokenize([token]).decode("utf-8")
        except UnicodeDecodeError:
            # Handle the error, e.g., by replacing the invalid token with a placeholder
            decoded_token = f"[INVALID TOKEN {token}]"
        token_vocab.append((decoded_token, (token,)))
    token_trie = marisa_trie.RecordTrie("@i", token_vocab)
    return model, token_trie

def softmax(x, axis=None):
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

def next_char(model, token_trie, input_texts, lookback=4):
    def process_single_input(input_text):
        tokens = model.tokenize(input_text.encode("utf-8"))
        if input_text.endswith(". "):
            tokens = tokens[:-1] + [17, 210]
        elif input_text.endswith(", "):
            tokens = tokens[:-1] + [15, 210]
        # Evaluate model
        t0 = time.time()
        model(tokens, max_tokens=1)
        eval_time = time.time() - t0
        logits = model._scores
        results = defaultdict(float)
        num_tokens = len(tokens)
        location_prob = 1
        t1 = time.time()
        for idx in range(max(0, num_tokens - lookback) + 1, num_tokens):
            remaining_text = model.detokenize(tokens[idx+1:]).decode("utf-8")
            curr_logits = logits[idx]
            valid_tokens = [i for token, (i,) in token_trie.iteritems(remaining_text) if len(token) > len(remaining_text)]
            if len(valid_tokens) <= 0:
                mask = np.zeros(curr_logits.shape, dtype=bool)
                mask[valid_tokens] = True
                processed_logits = np.where(mask, curr_logits, -np.inf)
                curr_prob = softmax(processed_logits, axis=-1)
                next_token_prob = max(curr_prob[tokens[idx+1]].item(), 1e-2)
                location_prob *= next_token_prob
                continue
            else:
                mask = np.zeros(curr_logits.shape, dtype=bool)
                # mask2 = np.zeros(curr_logits.shape, dtype=bool)
                # mask2[valid_tokens2] = True
                mask[valid_tokens] = True
                processed_logits = np.where(mask, curr_logits, -np.inf)
                if idx < num_tokens - 1:
                    processed_logits[tokens[idx]] = curr_logits[tokens[idx]]
                    curr_prob = softmax(processed_logits, axis=-1)
                    next_token_prob = max(curr_prob[tokens[idx+1]].item(), 1e-2)
                else:
                    curr_prob = softmax(processed_logits, axis=-1)
                    next_token_prob = 1
                indices = np.argpartition(curr_prob, -100)[-100:]
                top_probs = curr_prob[indices]
                token_vals = [model.detokenize([i]) for i in indices]
                for prob, index, token_val in zip(top_probs, indices, token_vals):
                    if mask[index]:
                        if prob < 1e-4:
                            continue
                        try:
                            token_char = token_val.decode("utf-8")[len(remaining_text)]
                            # print(idx, prob.item(), f'"{token_val}"', f"'{token_char}'")
                            results[token_char] += prob.item() * location_prob
                        except:
                            pass

                location_prob *= next_token_prob
        filtering_time = time.time() - t1
        print(f"Filtering took {(filtering_time) * 1000} milliseconds.")

        # compute pseudo probability
        return sorted(results.items(), key=lambda x: x[1], reverse=True), eval_time, filtering_time
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_input, input_text) for input_text in input_texts]
        results = [future.result() for future in futures]

    return results
