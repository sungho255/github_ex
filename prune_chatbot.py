import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune
import numpy as np

def magnitude_prune(model, amount):
    """
    Prunes the model's weights based on their magnitude.

    Args:
        model (torch.nn.Module): The model to prune.
        amount (float): The fraction of weights to prune (e.g., 0.2 for 20%).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

def calculate_sparsity(model):
    """
    Calculates the sparsity of the model.

    Args:
        model (torch.nn.Module): The model to check.

    Returns:
        float: The sparsity of the model.
    """
    total_zeros = 0
    total_params = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            total_params += p.nelement()
            total_zeros += torch.sum(p == 0)
    return total_zeros / total_params

def main():
    """
    Main function to demonstrate pruning a chatbot model.
    """
    try:
        # 1. Load a pre-trained chatbot model and tokenizer
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print(f"Loaded model: {model_name}")

        # 2. Define the pruning amount
        pruning_amount = 0.2  # Prune 20% of the weights
        print(f"Pruning {pruning_amount * 100}% of the weights.")

        # 3. Apply pruning
        magnitude_prune(model, pruning_amount)

        # 4. Calculate and print sparsity
        sparsity = calculate_sparsity(model)
        print(f"Model sparsity after pruning: {sparsity:.2%}")

        # 5. Demonstrate the pruned model with a simple chat
        print("\n--- Chat with the pruned model ---")
        print("Enter 'quit' to exit.")
        step = 0
        while True:
            user_input = input(">> User: ")
            if user_input.lower() == 'quit':
                break
            
            # Encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

            # Append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # Generate a response while limiting the total chat history to 1000 tokens
            chat_history_ids = model.generate(
                bot_input_ids, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id
            )

            # Pretty print last output
            print("Pruned DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
            step += 1

    except ImportError:
        print("Please install the required libraries: pip install torch transformers")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
