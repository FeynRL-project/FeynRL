import pandas as pd
import os

def create_rl_data(path):
    random_prompts = [
        {'prompt': [{"role": "user", "content": "What is 2+2?"}], 'solution': '4'},
        {'prompt': [{"role": "user", "content": "What is the capital of France?"}], 'solution': 'Paris'},
        {'prompt': [{"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "Why did the chicken cross the road?"}, {"role": "user", "content": "I don't know, why?"}], 'solution': 'To get to the other side!'},
        {'prompt': [{"role": "user", "content": "Write a poem about AI."}], 'solution': 'Code and logic, deep and wide, silicon mind where thoughts reside.'},
    ]
    df = pd.DataFrame(random_prompts)
    df.to_parquet(path, index=False)
    print(f"Created RL dummy data at {path}")

def create_cl_data(path):
    random_preferences = [
        {
            'prompt': [{"role": "user", "content": "What is 2+2?"}],
            'answer': "The answer is 4.",
            'rejected_answer': "The answer is 5."
        },
        {
            'prompt': [{"role": "user", "content": "Hello!"}],
            'answer': "Hi there! How can I help you today?",
            'rejected_answer': "Go away."
        },
        {
            'prompt': [{"role": "user", "content": "Tell me a story."}],
            'answer': "Once upon a time, there was a tiny robot.",
            'rejected_answer': "No."
        },
        {
            'prompt': [{"role": "user", "content": "What is AI?"}],
            'answer': "AI stands for Artificial Intelligence.",
            'rejected_answer': "AI is magic bacon."
        }
    ]
    df = pd.DataFrame(random_preferences)
    df.to_parquet(path, index=False)
    print(f"Created CL dummy data at {path}")

def create_sl_data(path):
    random_pairs = [
        {
            'prompt': [{"role": "user", "content": "Explain gravity."}],
            'answer': "Gravity is a force that pulls objects toward each other."
        },
        {
            'prompt': [{"role": "user", "content": "What is the capital of Japan?"}],
            'answer': "Tokyo."
        },
        {
            'prompt': [{"role": "user", "content": "Translate 'Hello' to Spanish."}],
            'answer': "Hola."
        },
        {
            'prompt': [{"role": "user", "content": "What is 5*5?"}],
            'answer': "25."
        }
    ]
    df = pd.DataFrame(random_pairs)
    df.to_parquet(path, index=False)
    print(f"Created SL dummy data at {path}")

if __name__ == "__main__":
    os.makedirs("launch_tests/data", exist_ok=True)
    create_rl_data("launch_tests/data/train_rl.parquet")
    create_rl_data("launch_tests/data/val_rl.parquet")
    create_cl_data("launch_tests/data/train_cl.parquet")
    create_cl_data("launch_tests/data/val_cl.parquet")
    create_sl_data("launch_tests/data/train_sl.parquet")
    create_sl_data("launch_tests/data/val_sl.parquet")
