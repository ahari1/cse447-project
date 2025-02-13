from collections import defaultdict
import numpy as np
import os
import subprocess
import sys
import marisa_trie

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
    filename = "Llama-3.2-1B.Q5_K_M.gguf"
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)

def next_char(model, token_trie, input_text, lookback=4):
    tokens = model.tokenize(input_text.encode("utf-8"))
    # Evaluate model
    model(tokens, max_tokens=1)
    logits = model._scores
    results = defaultdict(float)
    num_tokens = len(tokens)
    location_prob = 1
    for idx in range(max(0, num_tokens - lookback), num_tokens):
        remaining_text = model.detokenize(tokens[idx+1:]).decode("utf-8")
        curr_logits = logits[idx]
        valid_tokens = [i for token, (i,) in token_trie.iteritems(remaining_text) if len(token) > len(remaining_text)]
        # valid_tokens2 = [token_trie[token] for token in token_trie.prefixes(remaining_text)]
        # valid_tokens = [i for i, token in enumerate(token_vocab) if len(token) > len(remaining_text) and token.startswith(remaining_text)]
        # valid_tokens2 = [i for i, token in enumerate(token_vocab) if len(token) <= len(remaining_text) and remaining_text.startswith(token)]
        if len(valid_tokens) <= 0:
            mask = np.zeros(curr_logits.shape, dtype=bool)
            # mask[valid_tokens2] = True
            mask[valid_tokens] = True
            processed_logits = np.where(mask, curr_logits, -np.inf)
            processed_logits[tokens[idx+1]] = curr_logits[tokens[idx+1]]
            curr_prob = softmax(processed_logits)
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
                processed_logits[tokens[idx+1]] = curr_logits[tokens[idx+1]]
                curr_prob = softmax(processed_logits)
                next_token_prob = max(curr_prob[tokens[idx+1]].item(), 1e-2)
            else:
                curr_prob = softmax(processed_logits)
                next_token_prob = 1
            indices = np.argpartition(curr_prob, -100)[-100:]
            top_probs = curr_prob[indices]
            for prob, index in zip(top_probs, indices):
                if prob < 1e-4:
                    continue
                if mask[index]:
                    try:
                        token_val = model.detokenize([index])
                        token_char = token_val.decode("utf-8")[len(remaining_text)]
                        # print(idx, prob.item(), f'"{token_val}"', f"'{token_char}'")
                        results[token_char] += prob.item() * location_prob
                    except:
                        pass

            location_prob *= next_token_prob

    # compute pseudo probability
    return sorted(results.items(), key=lambda x: x[1], reverse=True)
