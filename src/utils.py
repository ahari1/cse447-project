from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from collections import defaultdict
import sys

def load_bloom(work_dir="../work", device="cpu"):
    # Load model and tokenizer
    # https://huggingface.co/docs/transformers/en/gguf
    model_path = sys.path.append(work_dir, "bloom-560m.q8_0.gguf")
    model_id = "afrideva/bloom-560m-GGUF"
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=model_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=model_path)

    token_vocab = tokenizer.batch_decode(torch.arange(tokenizer.vocab_size))

    model.to(device)
    return model, tokenizer, token_vocab


def next_char(model, tokenizer, token_vocab, input_text, lookback=4, device="cpu") -> list[tuple[str, float]]:
    tokens = tokenizer(input_text, return_tensors="pt")
    if tokens["input_ids"].shape[1] < 2:
        # If input size is too small, just feed model with \n + tokens
        tokens["input_ids"] = torch.cat([torch.tensor([[189]], dtype=int), tokens["input_ids"]], dim=1)
        tokens["attention_mask"] = torch.cat([torch.tensor([[1]], dtype=int), tokens["attention_mask"]], dim=1)
    
    if input_text.endswith(". "):
        # If the text ends with ". ", tokenize as "." and " " instead of ". "
        tokens["input_ids"] = torch.cat([tokens["input_ids"][:, :-1], torch.tensor([[17, 210]], dtype=int)], dim=1)
        tokens["attention_mask"] = torch.cat([torch.tensor([[1]], dtype=int), tokens["attention_mask"]], dim=1)
    elif input_text.endswith(", "):
        tokens["input_ids"] = torch.cat([tokens["input_ids"][:, :-1], torch.tensor([[15, 210]], dtype=int)], dim=1)
        tokens["attention_mask"] = torch.cat([torch.tensor([[1]], dtype=int), tokens["attention_mask"]], dim=1)

    # potentially add in later
    # elif input_text.endswith( char for char in string.punctuation + " "):
   
    tokens["input_ids"].to(device)
    tokens["attention_mask"].to(device)
    # Evaluate model
    with torch.no_grad():
        outputs = model(**tokens) # Changed from inputs to tokens
        logits = outputs.logits
    # we want to return ALL logits. This allows us to try different positions
    # I am an astronaut and 
    # I am an astronaut P(anderson) / P(all tokens that even partially complete)
    logits = logits[0, :, :].to("cpu")
    results = defaultdict(float)
    num_tokens = tokens["input_ids"].shape[1]
    location_prob = 1
    for idx in range(max(0, num_tokens - lookback) + 1, num_tokens):
        remaining_text = tokenizer.decode(tokens["input_ids"][0, idx+1:])
        curr_logits = logits[idx].clone() # predict next token for input ending at token idx
        valid_tokens = [i for i, token in enumerate(token_vocab) if len(token) > len(remaining_text) and token.startswith(remaining_text)]
        valid_tokens2 = [i for i, token in enumerate(token_vocab) if len(token) <= len(remaining_text) and remaining_text.startswith(token)]
        if len(valid_tokens) <= 0:
            # no new tokens to add to predcitions
            mask = torch.zeros(curr_logits.shape, dtype=bool)
            mask[valid_tokens2] = True
            mask[valid_tokens] = True
            processed_logits = torch.where(mask, curr_logits, -torch.inf)
            curr_prob = F.softmax(processed_logits, dim=-1)
            next_token_prob = max(curr_prob[tokens["input_ids"][0, idx]].item(), 1e-2) # try idx+1 here?
            location_prob *= next_token_prob
            continue
        else:
            # we found candidates
            mask = torch.zeros(curr_logits.shape, dtype=bool)
            mask2 = torch.zeros(curr_logits.shape, dtype=bool)
            mask2[valid_tokens2] = True
            mask[valid_tokens] = True
            processed_logits = torch.where(mask | mask2, curr_logits, -torch.inf)
            if idx < num_tokens - 1:
                # not at the end
                processed_logits[tokens["input_ids"][0, idx]] = curr_logits[tokens["input_ids"][0, idx]] # try +1???
                curr_prob = F.softmax(processed_logits, dim=-1)
                next_token_prob = max(curr_prob[tokens["input_ids"][0, idx]].item(), 1e-2)
            else:
                # predicting starting from last token
                curr_prob = F.softmax(processed_logits, dim=-1)
                next_token_prob = 1
            top_probs, indices = torch.topk(curr_prob, 100)
            token_vals = tokenizer.batch_decode(indices)
            for prob, index, token_val in zip(top_probs, indices, token_vals):
                if mask[index]:
                    # completely completes remaining text
                    if prob < 1e-4:
                        continue
                    token_char = token_val[len(remaining_text)]
                    # I am an astronaut and
                    # prob.item() = P(anderson | I am an astronaut )
                    # location_prob = P(I am an astronaut)
                    # location_prob = P(and | I am an astronaut ) ?
                    # What we want: P(anderson | I am an astronaut and) * P(I am an astronaut and)

                    
                    results[token_char] += prob.item() * location_prob # /?

            location_prob *= next_token_prob

    # compute pseudo probability
    return sorted(results.items(), key=lambda x: x[1], reverse=True)