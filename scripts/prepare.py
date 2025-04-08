from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-pt")
breakpoint()
