from collections import defaultdict
import numpy as np
import os
import subprocess
import sys
import marisa_trie
import time
from multiprocessing import Pool, shared_memory

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
    decoded_tokens = []
    for token in range(NUM_TOKENS):
        token_bytes = model.detokenize([token])
        try:
            decoded_token = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Handle the error, e.g., by replacing the invalid token with a placeholder
            decoded_token = f"."
        decoded_tokens.append(decoded_token)
        token_vocab.append(token_bytes)
    token_trie = marisa_trie.RecordTrie("@i", [(token, (i,)) for i, token in enumerate(decoded_tokens)])
    return model, token_vocab, token_trie

def init_worker(token_vocab, token_trie):
    """Function used to initialize shared data for workers"""
    global vocab
    global trie
    trie = token_trie
    vocab = token_vocab

def init_pool(token_vocab, token_trie):
    """Loads the multiprocessing pool"""
    pool = Pool(initializer=init_worker, initargs=(token_vocab,token_trie), processes=4)
    return pool

def softmax(x, axis=None):
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x

def filtering_step(logit_info, tokens, input_text, lookback):
    results = defaultdict(float)
    num_tokens = len(tokens)
    location_prob = 1
    shm_name, shape, dtype = logit_info
    shm = shared_memory.SharedMemory(name=shm_name)
    logits = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    t1 = time.time()
    start_pos = max(0, num_tokens - lookback)
    for idx in range(start_pos, num_tokens):
        try:
            remaining_text = b''.join([vocab[tokens[i]] for i in range(idx+1, len(tokens))]).decode("utf-8")
        except:
            # just skip
            print("ERROR")
            continue
        curr_logits = logits[idx - start_pos]
        valid_tokens = [i for token, (i,) in trie.items(remaining_text) if len(token) > len(remaining_text)]
        if len(valid_tokens) <= 0:
            location_prob *= 1e-2
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
            token_vals = [vocab[i] for i in indices]
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

    # delete shared memory
    shm.close()
    # compute pseudo probability
    return sorted(results.items(), key=lambda x: x[1], reverse=True), filtering_time

def next_char(model, pool, token_vocab, token_trie, input_texts, lookback=4):
    eval_times = []
    pool_results = []
    shms = []
    for input_text in input_texts:
        # Run model
        tokens = model.tokenize(input_text.encode("utf-8"))
        if input_text.endswith(". "):
            tokens = tokens[:-1] + [17, 210]
        elif input_text.endswith(", "):
            tokens = tokens[:-1] + [15, 210]
        t0 = time.time()
        model(tokens, max_tokens=1)
        eval_time = time.time() - t0
        eval_times.append(eval_time)

        # use shared memory for greater speed
        logits = model._scores[-lookback:]
        shm = shared_memory.SharedMemory(create=True, size=logits.nbytes)
        shm_logits = np.ndarray(logits.shape, dtype=logits.dtype, buffer=shm.buf)
        np.copyto(shm_logits, logits)

        # run filtering in separate process
        pool_results.append(pool.apply_async(filtering_step, ((shm.name, shm_logits.shape, shm_logits.dtype), tokens, input_text, lookback)))

        shms.append(shm)

    results = []
    # gather results from pool
    for eval_time, pool_result, shm in zip(eval_times, pool_results, shms):
        preds, filtering_time = pool_result.get()
        results.append((preds, eval_time, filtering_time))
        # close shared memory object
        shm.close()
        shm.unlink()
    return results
