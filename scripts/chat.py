import sys
import atexit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "out/qwen0.5b-qat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Loading {model_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params). Type 'exit' to quit.\n")

    history = []
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in ("exit", "quit", "q"):
            break
        if not prompt:
            continue

        history.append({"role": "user", "content": prompt})

        messages = []
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_part = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        history.append({"role": "assistant", "content": assistant_part})
        print(assistant_part)


if __name__ == "__main__":
    main()
