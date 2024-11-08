import torch
from myTransformer.cache import prepare_cache_for_generation
from transformers.generation.configuration_utils import GenerationConfig

if __name__ == "__main__":
    import transformers
    from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
    # i = int(sys.argv[1])
    device = "cuda:0"

    transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation

    model_path = "/nfs/shared_LLM_model/lmsys/longchat-7b-v1.5-32k"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "sdpa"

    print(config)

    model = LlamaForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             config=config)
    model.generation_config.cache_implementation = "static"
    model = model.eval().to(device)

    batch_size = 2
    print(f"batch_size: {batch_size}")
    input_text = "A " * 1000

    prompt = tokenizer.encode(input_text, return_tensors="pt")
    prompt = prompt.to(model.device)
    prompt = prompt.repeat(batch_size, 1)
    generation_kwargs = {
        "page_num": 1000,
        "page_size": 16,
    }
    generation_config = GenerationConfig(**generation_kwargs)
    model.generate(prompt,
                   max_new_tokens=1,
                   min_new_tokens=1,
                   generation_config=generation_config)
    kvcache = model._cache

    key_states = torch.randn((batch_size, 1024, 32, 128),
                             dtype=torch.float16,
                             device=device)
    value_states = torch.randn((batch_size, 1024, 32, 128),
                               dtype=torch.float16,
                               device=device)
    kvcache.update(key_states, value_states, 0)
    print(kvcache.layer_allocators[0].req2page)
    print(kvcache.layer_allocators[0].get_metadata([0, 1]))

    key_states = torch.randn((batch_size, 1, 32, 128),
                             dtype=torch.float16,
                             device=device)
    value_states = torch.randn((batch_size, 1, 32, 128),
                               dtype=torch.float16,
                               device=device)
    kvcache.update(key_states, value_states, 0)
    print(kvcache.layer_allocators[0].get_metadata([0, 1]))

    print("Reset")
    model.generate(prompt,
                   max_new_tokens=1,
                   min_new_tokens=1,
                   generation_config=generation_config)

    print(kvcache.layer_allocators[0].req2page)

    key_states = torch.randn((batch_size, 512, 32, 128),
                             dtype=torch.float16,
                             device=device)
    value_states = torch.randn((batch_size, 512, 32, 128),
                               dtype=torch.float16,
                               device=device)
    kvcache.update(key_states, value_states, 0)
    print(kvcache.layer_allocators[0].get_metadata([0, 1]))

    key_states = torch.randn((batch_size, 1, 32, 128),
                             dtype=torch.float16,
                             device=device)
    value_states = torch.randn((batch_size, 1, 32, 128),
                               dtype=torch.float16,
                               device=device)
    kvcache.update(key_states, value_states, 0)
    print(kvcache.layer_allocators[0].get_metadata([0, 1]))

    key_states = torch.randn((batch_size, 1, 32, 128),
                             dtype=torch.float16,
                             device=device)
    value_states = torch.randn((batch_size, 1, 32, 128),
                               dtype=torch.float16,
                               device=device)
    kvcache.update(key_states, value_states, 0)
    print(kvcache.layer_allocators[0].get_metadata([0, 1]))
