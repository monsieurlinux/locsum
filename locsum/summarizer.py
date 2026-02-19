# Standard library imports
import time

# Third-party library imports
import ollama

# Local imports
from colors import GREEN, YELLOW, RED, RESET
from logger import logger
from utils import format_time


def summarize(transcript, model, config):
    # Summarize with Ollama
    CONFIG = config

    print(f'Summarizing with {YELLOW}{model}{RESET} model')
    start_time = time.time()

    # Initialize the conversation list with system + user prompts
    messages = [{
        "role": "system",
        "content": CONFIG['summary']['system_prompt']
    }]

    messages.append({
        "role": "user",
        "content": f"{CONFIG['summary']['user_prompt']}\n\n{transcript}"
    })

    # Get the first response
    response = ollama.chat(model=model, messages=messages)
    summary = response['message']['content']
    exec_time = time.time() - start_time
    ratio_pct = len(summary) / len(transcript) * 100
    logger.debug(f'Done in {format_time(exec_time)} ({ratio_pct:.1f}% ratio)')

    # Add the response to conversation history
    messages.append({
        "role": "assistant",
        "content": summary
    })
    
    # Determine the summary target ratio based on transcript size
    transcript_size = len(transcript)
    
    if transcript_size < CONFIG['summary']['small_transcript_max_size']:
        target_ratio = CONFIG['summary']['small_transcript_target_ratio']
    elif transcript_size < CONFIG['summary']['medium_transcript_max_size']:
        target_ratio = CONFIG['summary']['medium_transcript_target_ratio']
    else:
        target_ratio = CONFIG['summary']['large_transcript_target_ratio']

    # Request details if summary too short
    if ratio_pct < target_ratio:
        # TODO: Maybe replace by while loop with max number of iterations
        print(f"Summary is too short ({RED}{ratio_pct:.1f}%{RESET} ratio for "
              f"{GREEN}{target_ratio}%{RESET} target), asking for more details")
        start_time = time.time()
        
        # Add the prompt to request a more detailed summary
        messages.append({
            "role": "user", 
            "content": CONFIG['summary']['expand_prompt']
        })
        
        # Get the new response
        response = ollama.chat(model=model, messages=messages)
        summary = response['message']['content']
        exec_time = time.time() - start_time
        ratio_pct = len(summary) / len(transcript) * 100
        logger.debug(f'Done in {format_time(exec_time)} ({ratio_pct:.1f}% ratio)')
        
        color = RED if ratio_pct < target_ratio else GREEN
        print(f"New summary has a {color}{ratio_pct:.1f}%{RESET} ratio")
        
        # Add the response to conversation history for the next iteration
        messages.append({
            "role": "assistant",
            "content": summary
        })

    return summary


def is_model_available(model_name: str) -> bool:
    # Fetch local models
    models = ollama.list()['models']

    # Extract just the names into a list
    names = [m['model'] for m in models]

    # Check for exact match or with 'latest' suffix
    return model_name in names or f'{model_name}:latest' in names


def get_context_length(model_name: str) -> int:
    try:
        modelinfo = ollama.show(model_name).get("modelinfo")

        if not isinstance(modelinfo, dict):
            logger.debug(f"'modelinfo' not found or not a dict for model '{model_name}'")
            return 0

        # Look for any key ending with '.context_length'
        for key, value in modelinfo.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    logger.debug(f"Context length value for key '{key}' is not an integer: {value}")
                    continue

        logger.debug(f"No '.context_length' key found in modelinfo for '{model_name}'")
        return 0

    except Exception as e:
        logger.debug(f"Error fetching model info for '{model_name}': {e}")
        return 0


def test_model_speed(transcript_text):
    runs = 10
    times_q4km = []
    times_q8_0 = []
    times_bf16 = []

    for i in range(runs):
        model = 'glm-4.7-flash'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model)
        end = time.perf_counter()
        times_q4km.append(end - start)

        model = 'glm-4.7-flash:q8_0'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model)
        end = time.perf_counter()
        times_q8_0.append(end - start)

        model = 'glm-4.7-flash:bf16'
        print(f'Run {i} with model {model}')
        start = time.perf_counter()
        summarize(transcript_text, model)
        end = time.perf_counter()
        times_bf16.append(end - start)

        avg_q4km = sum(times_q4km) / len(times_q4km)
        avg_q8_0 = sum(times_q8_0) / len(times_q8_0)
        avg_bf16 = sum(times_bf16) / len(times_bf16)

        print(f"Average time for q4km: {avg_q4km} seconds")  # 69.1 sec
        print(f"Average time for q8_0: {avg_q8_0} seconds")  # 86.6 sec (+25%)
        print(f"Average time for bf16: {avg_bf16} seconds")  # 132.4 sec (+92%)


