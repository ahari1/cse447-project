from collections import defaultdict
import numpy as np
import os
import subprocess
import sys
import marisa_ext
import time
from multiprocessing import Pool

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

def load_model(work_dir):
    # Load model and tokenizer
    # https://huggingface.co/docs/transformers/en/gguf
    Llama, use_gpu = get_llama()
    filename = "Llama-3.2-1B.Q5_K_M.gguf"
    model = Llama(model_path=os.path.join(work_dir, filename), logits_all=True, n_gpu_layers=-1 if use_gpu else 0, verbose=False)

    # load token vocabulary
    NUM_TOKENS = model.n_vocab()
    token_vocab = []
    decoded_tokens = []
    for token in range(NUM_TOKENS):
        token_bytes = model.detokenize([token])
        try:
            decoded_token = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Handle the error, e.g., by replacing the invalid token with a placeholder
            decoded_token = ""
        decoded_tokens.append(decoded_token)
        token_vocab.append(token_bytes)

    token_trie = marisa_ext.FastTrie(decoded_tokens)
    return model, token_trie, token_vocab

def init_worker(work_dir):
    """Function used to initialize shared data for workers"""
    global trie, vocab, model
    model, trie, vocab = load_model(work_dir)

def init_pool(work_dir="../work"):
    """Loads the multiprocessing pool"""
    pool = Pool(initializer=init_worker, initargs=(work_dir,), processes=4)
    return pool

def softmax(x, axis=None):
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

def process_input(input_text, lookback):
    tokens = model.tokenize(input_text.encode("utf-8"))
    model(tokens, max_tokens=1)
    results = defaultdict(float)
    num_tokens = len(tokens)
    location_prob = 1
    logits = model._scores[-lookback:]
    start_pos = max(0, num_tokens - lookback)
    for idx in range(start_pos, num_tokens):
        try:
            remaining_text = b''.join([vocab[tokens[i]] for i in range(idx+1, len(tokens))])
        except:
            # just skip
            print("ERROR")
            continue
        curr_logits = logits[idx - start_pos]
        # valid_tokens = [i for token, (i,) in trie.items(remaining_text) if len(token) > len(remaining_text)]
        if remaining_text:
            valid_tokens = trie.get_valid_tokens(remaining_text)
        else:
            valid_tokens = np.arange(len(vocab))
        if len(valid_tokens) <= 0:
            location_prob *= 1e-2
            continue
        else:
            # mask = np.zeros(curr_logits.shape, dtype=bool)
            # mask2 = np.zeros(curr_logits.shape, dtype=bool)
            # mask2[valid_tokens2] = True
            # mask[valid_tokens] = True
            # processed_logits = np.where(mask, curr_logits, -np.inf)
            processed_logits = curr_logits[valid_tokens]
            if idx < num_tokens - 1:
                # processed_logits[tokens[idx]] = curr_logits[tokens[idx]]
                next_token_logit = curr_logits[tokens[idx+1]]
                max_val = max(np.max(processed_logits, keepdims=True), next_token_logit)
                exp_x = np.exp(processed_logits - max_val)
                exp_next = np.exp(next_token_logit - max_val)
                sum_exp_x = np.sum(exp_x, keepdims=True) + exp_next
                curr_prob = exp_x / sum_exp_x
                next_token_prob = max((exp_next / sum_exp_x).item(), 1e-2)
            else:
                curr_prob = softmax(processed_logits, axis=-1)
                next_token_prob = 1
            indices = np.argpartition(curr_prob, max(-100, -len(valid_tokens)))[-100:]
            top_probs = curr_prob[indices]
            orig_indices = valid_tokens[indices]
            token_vals = [vocab[i] for i in orig_indices]
            for prob, index, token_val in zip(top_probs, orig_indices, token_vals):
                if prob < 1e-4:
                    continue
                try:
                    token_char = token_val.decode("utf-8")[len(remaining_text)]
                    # print(idx, prob.item(), f'"{token_val}"', f"'{token_char}'")
                    results[token_char] += prob.item() * location_prob
                except:
                    pass

            location_prob *= next_token_prob
    # compute pseudo probability
    return sorted(results.items(), key=lambda x: x[1], reverse=True)

def next_char(pool, input_texts, lookback=4):
    pool_results = []
    for input_text in input_texts:
        pool_results.append(pool.apply_async(process_input, (input_text, lookback)))

    results = []
    # gather results from pool
    for pool_result in pool_results:
        preds = pool_result.get()
        results.append(preds)
    return results
