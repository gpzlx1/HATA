# HATA: Hash-Aware Top-k Attention

Large Language Models (LLMs) have emerged as a pivotal research area, yet the attention module remains a critical bottleneck in LLM inference, even with techniques like KVCache to mitigate redundant computations. While various top-k attention mechanisms have been proposed to accelerate LLM inference by exploiting the inherent sparsity of attention, they often struggled to strike a balance between efficiency and accuracy. In this paper, we introduce HATA (Hash-Aware Top-k Attention), a novel approach that systematically integrates low-overhead learning-to-hash techniques into the Top-k attention process. Different from the existing top-k attention methods which are devoted to seeking an absolute estimation of qk score, typically with a great cost, HATA maps queries and keys into binary hash codes, and acquires the relative qk score order with a quite low cost, which is sufficient for realizing top-k attention. Extensive experiments demonstrate that HATA achieves up to 7.2× speedup compared to vanilla full attention while maintaining model accuracy. In addition, HATA outperforms the state-of-the-art top-k attention methods in both accuracy and efficiency across multiple mainstream LLM models and diverse tasks.



## Install

* CUDA >= 12.4
* Transfomers == 4.47.1
* torch == 2.4.0

### Install Dependencies

```shell
pip install -r requirements.txt
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4
```

### Install

```shell
bash install.sh
```

## Run

```shell
cd tasks
bash run_*.sh
```

Currently, the repo is still required to be refactored. It will be finished in 1-2 weeks.

## Citation

```shell
@inproceedings{gong-etal-2025-hata,
    title = "{HATA}: Trainable and Hardware-Efficient Hash-Aware Top-$k$ Attention for Scalable Large Model Inference",
    author = "Gong, Ping  and
      Yi, Jiawei  and
      Wang, Shengnan  and
      Zhang, Juncheng  and
      Jin, Zewen  and
      Zhou, Ouxiang  and
      Liu, Ruibo  and
      Xu, Guanbin  and
      Bai, Youhui  and
      Ye, Bowen  and
      Yuan, Kun  and
      Yang, Tong  and
      Zhang, Gong  and
      Chen, Renhai  and
      Wu, Feng  and
      Li, Cheng",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1275/",
    pages = "24856--24871",
    ISBN = "979-8-89176-256-5"
}
```

