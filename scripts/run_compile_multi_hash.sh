#!/bin/bash

############################### compile for decode_hash_encode ###############################
python3 -m triton.tools.compile \
     -n "_decode_multi_hash_encode" \
     -w 4 \
     -ns 1 \
     -on "_decode_multi_hash_encode_rbit256_dim128_kernel" \
     -o "decode_multi_hash_encode_rbit256_dim128" \
     -s "*fp16,i64, *fp16,i64, *fp16,*i32, *i32,i64, *fp16,i64, *i32,i64, i32,i32,i32,i32, 256,128,16" \
     -g 2,2,8 ./python/myTransformer/cache/kernels/triton_hash_encode.py

sleep 3
mv decode_multi_hash_encode_rbit256_dim128*.c ./src/decode_multi_hash_encode_rbit256_dim128.c
mv decode_multi_hash_encode_rbit256_dim128*.h ./src/decode_multi_hash_encode_rbit256_dim128.h

# python3 -m triton.tools.compile \
#      -n "_decode_hash_encode" \
#      -w 4 \
#      -ns 1 \
#      -on "_decode_hash_encode_rbit256_dim128_kernel" \
#      -o "decode_hash_encode_rbit256_dim128" \
#      -s "*fp16,i64, *fp16,i64, *fp16,*i32, *i32,i64, *fp16,i64, *i32,i64, i32,i32,i32,i32, 256,128,16" \
#      -g 4,1,8 ./python/myTransformer/cache/kernels/triton_hash_encode.py

# sleep 3
# mv decode_hash_encode_rbit256_dim128*.c ./src/decode_hash_encode_rbit256_dim128.c
# mv decode_hash_encode_rbit256_dim128*.h ./src/decode_hash_encode_rbit256_dim128.h


############################### compile for prefill_hash_encode ###############################
# python3 -m triton.tools.compile \
#      -n "_prefill_hash_encode" \
#      -w 4 \
#      -ns 2 \
#      -on "_prefill_hash_encode_rbit128_dim128_kernel" \
#      -o "prefill_hash_encode_rbit128_dim128" \
#      -s "*fp16,i64, *fp16,*i32, *i32,i64, *fp16,i64, i32,i32, 128,128,128" \
#      -g 2500,1,1 ./python/myTransformer/cache/kernels/triton_hash_encode.py

# sleep 3
# mv prefill_hash_encode_rbit128_dim128*.c ./src/prefill_hash_encode_rbit128_dim128.c
# mv prefill_hash_encode_rbit128_dim128*.h ./src/prefill_hash_encode_rbit128_dim128.h

# python3 -m triton.tools.compile \
#      -n "_prefill_hash_encode" \
#      -w 4 \
#      -ns 2 \
#      -on "_prefill_hash_encode_rbit256_dim128_kernel" \
#      -o "prefill_hash_encode_rbit256_dim128" \
#      -s "*fp16,i64, *fp16,*i32, *i32,i64, *fp16,i64, i32,i32, 256,128,128" \
#      -g 2500,1,1 ./python/myTransformer/cache/kernels/triton_hash_encode.py

# sleep 3
# mv prefill_hash_encode_rbit256_dim128*.c ./src/prefill_hash_encode_rbit256_dim128.c
# mv prefill_hash_encode_rbit256_dim128*.h ./src/prefill_hash_encode_rbit256_dim128.h

############################### compile for decode_hash_encode ###############################
# python3 -m triton.tools.compile \
#      -n "_combine_attention" \
#      -w 4 \
#      -ns 1 \
#      -on "_combine_attention_dim128_kernel" \
#      -o "combine_attention_dim128" \
#      -s "*fp16, *fp16, *fp32, *fp32, *fp16, i32, 128, 8" \
#      -g 32,16,1 ./python/myTransformer/cache/kernels/triton_combine_attn.py

# sleep 3
# mv combine_attention_dim128*.c ./src/combine_attention_dim128.c
# mv combine_attention_dim128*.h ./src/combine_attention_dim128.h
