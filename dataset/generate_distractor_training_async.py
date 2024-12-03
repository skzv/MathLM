import json
import asyncio
import time
from openai import AsyncOpenAI, RateLimitError

client = AsyncOpenAI(api_key='sk-proj')

# Semaphore to limit concurrent API calls
semaphore = asyncio.Semaphore(20)

async def prompt_openai_api(perturbation_name, perturbation_description, seed_question, answer_rationale):
    user_prompt = f"""
    Your objective is to rewrite a given math question using the specified perturbation strategy ({perturbation_name}). The rewritten question should be reasonable, understandable, and able to be responded to by humans.

    Perturbation strategy: {perturbation_description}

    The given question: {seed_question}

    Answer of the given question: {answer_rationale}
    
    Please rewrite the question using the specified perturbation strategies while minimizing edits to avoid significant
    deviation in the question content. It is important to ensure that the rewritten question has only one required
    numerical answer.
    
    The rewritten question:
    """
    
    system_message = "You are a helpful assistant, good at following instructions."

    # Implement retry logic with exponential backoff
    attempt = 0
    while attempt < 5:  # Retry up to 5 times
        try:
            # Acquire the semaphore before making the request
            async with semaphore:
                completion = await client.chat.completions.create(
                    model="gpt-4o",  # Use the appropriate model
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ]
                )
            rewritten_question = completion.choices[0].message.content
            return rewritten_question
        except RateLimitError as e:
            attempt += 1
            delay = 2 ** attempt  # Exponential backoff (1, 2, 4, 8, ...)
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)  # Sleep before retrying
        except Exception as e:
            print(f"Error during request: {e}")
            break  # Break on any other error
    return None  # Return None if the maximum retries are exceeded

async def process_data(input_file, output_file):
    # Read the data from the input file
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    tasks = []
    for i, line in enumerate(data):
        if i % 100 == 0:  # Print progress every 100 lines
            print(f"Processing line {i}...")
        
        seed_question = line['question']
        answer_rationale = line['answer']

        # Create an async task for each API call
        task = prompt_openai_api(perturbation_name, perturbation_description, seed_question, answer_rationale)
        tasks.append(task)

    # Wait for all tasks to complete concurrently, respecting the semaphore limit
    rewritten_questions = await asyncio.gather(*tasks)

    # Assign the rewritten questions to the respective entries
    for i, rewritten_question in enumerate(rewritten_questions):
        if rewritten_question:  # Only assign if the question was successfully rewritten
            data[i]['rewritten_question'] = rewritten_question

    # Write the processed data to the output file
    with open(output_file, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')  # Add a newline after each JSON object

# Define your perturbation details
perturbation_name = "Distractor Insertion "
perturbation_description = "involves introducing distracting conditions that have no impact on the final answer. These introduced conditions should be relevant to the topic of the original question and preferably include numerical values. However, the rewritten problem must maintain an identical solution to that of the original problem."

# Define input and output file paths
input_file = './gsm8k/original_train.jsonl'
output_file = 'gsm_train_with_rewritten.jsonl'

# Run the async process
asyncio.run(process_data(input_file, output_file))
