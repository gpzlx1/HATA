import numpy as np
import json
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import random

RANDOM_NEEDLE_CITIES = [
    "Chicago", "Yangon", "Antananarivo", "Colombo", "Almaty", "Sydney",
    "Chicago", "Mexico City", "Seattle", "Lagos", "Amsterdam", "Belgrade",
    "Cairo", "Baghdad", "Damascus", "Kigali", "Dakar", "Dakar", "Sofia",
    "Kigali", "Victoria", "Tashkent", "Mumbai", "Barcelona", "Almaty", "Amman",
    "Toronto", "Bratislava", "Johannesburg", "Thimphu", "Bangkok", "Santiago",
    "Cairo", "San Francisco", "Lagos", "Amsterdam", "Paris", "Rabat",
    "Santiago", "Copenhagen", "Madrid", "Kigali", "Ho Chi Minh City",
    "Sarajevo", "Delhi", "Istanbul", "Ho Chi Minh City", "Khartoum",
    "Helsinki", "Doha", "Istanbul", "Kuala Lumpur", "Budapest", "Shanghai",
    "Moscow", "Los Angeles", "Oslo", "Johannesburg", "Berlin", "Bangalore",
    "Tokyo", "Melbourne", "Barcelona", "Chicago", "Port Louis", "Lisbon",
    "Nairobi", "Kampala", "Lima", "Maputo", "Vancouver", "Dubai", "Khartoum",
    "Jakarta", "Madrid", "Yerevan", "Beirut", "Athens", "Chicago", "Paris",
    "Bucharest", "Copenhagen", "Brussels", "Damascus", "Seattle",
    "Los Angeles", "Yerevan", "Victoria", "Tunis", "Astana", "Seoul",
    "Buenos Aires", "Bangkok", "Colombo", "Brussels", "Khartoum", "Doha",
    "San Francisco", "Vienna", "Jakarta"
]
NIAH_TEMPLATE = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\n\nQuestion: {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. Answer: "


def read_haystack(context_lengths, haystack_file, tokenizer):
    max_context_length = max(context_lengths)
    f = open(haystack_file, "r")
    context = ""
    toks = 0
    while toks < max_context_length:
        text = json.loads(f.readline())["text"]
        context += text
        toks += len(tokenizer.encode(text))
    return context


def generate_random_number(num_digits):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return random.randint(lower_bound, upper_bound)


def insert_needle_func(needle, context, depth_percent, context_length,
                       tokenizer, final_context_length_buffer):
    tokens_needle = tokenizer.encode(needle, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= final_context_length_buffer

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = [
            tokenizer.encode(".", add_special_tokens=False)[0],
            tokenizer.encode(". \n", add_special_tokens=False)[0],
            tokenizer.encode(".\n", add_special_tokens=False)[0],
            tokenizer.encode("\n", add_special_tokens=False)[0]
        ]

        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[
                -1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context,
                                   skip_special_tokens=True)
    return new_context


def create_context(
    id,
    trim_context,
    needle_number,
    needle_city,
    needle_template,
    question_template,
    context_length,
    depth_percent,
    tokenizer,
    final_context_length_buffer,
):
    needle = needle_template.format(city=needle_city, rnd_number=needle_number)
    question = question_template.format(needle_city)
    context = insert_needle_func(needle, trim_context, depth_percent,
                                 context_length, tokenizer,
                                 final_context_length_buffer)
    result = {
        "id": id,
        "context": context,
        "length": int(context_length),
        "depth_percent": float(depth_percent),
        "input": question,
        "answer": [needle_number],
        "key": needle_city,
        "language": "en",
        "dataset": "niah",
    }
    return result


if __name__ == "__main__":

    model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    haystack_file = f'./pg19_mini.jsonl'

    context_lengths_min = 1 * 1000
    context_lengths_max = 32 * 1000
    n_context_length_intervals = 15
    n_document_depth_intervals = 10  # position of the needle in the haystack

    needle = "\nThe special magic {city} number is: {rnd_number}\n"
    retrieval_question = "What is the special magic {} number?"
    rnd_number_digits = 7

    context_lengths = np.round(
        np.linspace(
            context_lengths_min,
            context_lengths_max,
            num=n_context_length_intervals,
            endpoint=True,
        )).astype(int)

    document_depth_percents = np.round(  # we use linear scale here
        np.linspace(
            0,
            100,
            num=n_document_depth_intervals,
            endpoint=True,
        )).astype(int)

    full_context = read_haystack(context_lengths=context_lengths,
                                 haystack_file=haystack_file,
                                 tokenizer=tokenizer)
    full_tokens = tokenizer.encode(full_context, add_special_tokens=False)

    tokenized_prompts = []
    gt = []
    ctx_len = []
    depth_pct = []

    id = 0
    f = open("niah.jsonl", mode="w+")

    for context_length in context_lengths:

        trim_context = tokenizer.decode(full_tokens[:context_length],
                                        skip_special_tokens=True)

        for depth_percent in document_depth_percents:

            needle_city = random.choice(RANDOM_NEEDLE_CITIES)
            needle_number = str(generate_random_number(rnd_number_digits))

            context = create_context(
                id=id,
                trim_context=trim_context,
                needle_number=needle_number,
                needle_city=needle_city,
                needle_template=needle,
                question_template=retrieval_question,
                context_length=context_length,
                depth_percent=depth_percent,
                tokenizer=tokenizer,
                final_context_length_buffer=32,
            )

            json.dump(context, f, ensure_ascii=False)
            f.write("\n")

            id += 1
