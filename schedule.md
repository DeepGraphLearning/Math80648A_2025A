---
layout: default
title: MATH 80648A - Machine Learning II<br>Deep Learning and Applications
permalink: /schedule

schedule:
  - date: Aug.<br>25
    topics:
    topics:
      - name: Introduction [<a href="assets/slides/Week1-Intro.pdf">En</a>]
      - name: Mathematics [<a href="assets/slides/Week1-Maths.pdf">En</a>]
      - name: Machine Learning Basics [<a href="assets/slides/Week1-ML.pdf">En</a>]
    readings:
      - name: Deep Learning Book
      - name: Chap. 2
        url: http://www.deeplearningbook.org/contents/linear_algebra.html
      - name: Chap. 3
        url: http://www.deeplearningbook.org/contents/prob.html
      - name: Chap. 5
        url: http://www.deeplearningbook.org/contents/ml.html
  - date: Sep.<br>8
    topics:
    - name: Feedforward Neural Networks & Optimization Tricks [<a href="assets/slides/Week2-FFN&Regularization.pdf">En</a>]
#      url: https://www.dropbox.com/s/zv4920r75ek7u4u/Week2-FFN%26Regularization.pdf?dl=0
    readings:
      - name: Deep Learning Book
      - name: Chap. 6
        url: http://www.deeplearningbook.org/contents/mlp.html
      - name: Chap. 7
        url: http://www.deeplearningbook.org/contents/regularization.html
      - name: Chap. 8
        url: http://www.deeplearningbook.org/contents/optimization.html
  - date: Sep.<br>15
    topics:
      - name: PyTorch part 1
#        url: https://www.dropbox.com/s/xpd4fjisk3n08vx/Deep%20Learning%20Frameworks%20part%201.pdf?dl=0
      - name: PyTorch Part 2
#        url: https://www.dropbox.com/s/2mzbdnfgah9yimw/Deep%20Learning%20Frameworks%20part%202.pdf?dl=0
    readings:
      - name: Python Numpy Tutorial
        url: http://cs231n.github.io/python-numpy-tutorial/
      - name: Neural Network from Scratch
        url: https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0
      - name: Dive into Deep Learning
        url: https://github.com/dsgiitr/d2l-pytorch
    homeworks:
      - name: HW1 (to be announced)
#      - name: Instruction
#        url: https://www.dropbox.com/s/7s19xya7yck4nul/HWK1.pdf?dl=0
#      - name: Colab
#        url: https://colab.research.google.com/drive/1FjRjNlBqPVz7SrEPvqrL10Q76NeHhvJW?usp=sharing
  - date: Sep.<br>22
    topics:
      - name: Convolutional Neural Networks & Recurrent Neural Networks
#        url: https://www.dropbox.com/s/9vnrcjo4ykhdj9l/Week4-CNN%26RNN.pdf?dl=0
    readings:
      - name: Deep Learning Book
      - name: Chap. 9
        url: http://www.deeplearningbook.org/contents/convnets.html
      - name: Chap. 10
        url: http://www.deeplearningbook.org/contents/rnn.html
    presentations:
      - name: ResNet
        url: http://arxiv.org/abs/1512.03385
      - name: GRU
        url: https://arxiv.org/abs/1412.3555
      - name: DenseNet
        url: https://arxiv.org/abs/1608.06993
      
      
  - date: Sep.<br>29
    topics:
      - name: Word Representation Learning
#        url: https://www.dropbox.com/s/bjxvre1d3w30iqj/Week5-DL4NLP-part1.pdf?dl=0
    readings:
      - name: Word2Vec
        url: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    presentations:
      - name: GloVe
        url: https://aclanthology.org/D14-1162.pdf
      - name: SGNS
        url: https://papers.nips.cc/paper_files/paper/2014/file/b78666971ceae55a8e87efb7cbfd9ad4-Paper.pdf
      
  - date: Oct.<br>6
    topics:
      - name: Attention, Transformers
#        url: https://www.dropbox.com/s/bjxvre1d3w30iqj/Week5-DL4NLP-part1.pdf?dl=0
    readings:
      - name: The annotated Transformer (blog)
        url: https://nlp.seas.harvard.edu/annotated-transformer/
      - name: Transformer
        url: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
      - name: Rotary Position Embedding
        url: https://arxiv.org/abs/2104.09864
    presentations:
      - name: Relative Position Embedding
        url: https://arxiv.org/abs/1803.02155
      - name: RoBERTa
        url: https://arxiv.org/abs/1907.11692
      - name: ViT
        url: https://arxiv.org/abs/2010.11929
      - name: Reformer
        url: https://arxiv.org/abs/2001.04451
      - name: FlashAttention
        url: https://arxiv.org/abs/2205.14135
  - date: Oct.<br>15
    topics:
      - name: No class (Project proposal)
    homeworks:
      - name: HW2 (to be announced)
#        url: https://www.dropbox.com/s/j2w4cpq14jypkbe/HW2.pdf?dl=0
#      - name: Instruction
#        url: https://www.dropbox.com/s/j2w4cpq14jypkbe/HW2.pdf?dl=0
#      - name: Kaggle
#        url: https://www.kaggle.com/c/math60630aw21
  - date: Nov.<br>3
    topics:
      - name: Large Language Models I
#        url: https://www.dropbox.com/s/366364m5gmu6gkd/Week7-DL4NLP-part2.pdf?dl=0
    readings:
      - name: BERT
        url: https://arxiv.org/pdf/1810.04805
      - name: GPT-1
        url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
      - name: GPT-3
        url: https://arxiv.org/abs/2005.14165
      - name: LoRA
        url: https://arxiv.org/abs/2104.09864
      - name: Scaling Law
        url: https://arxiv.org/abs/2001.08361
      - name: GPT in 60 Lines of NumPy
        url: https://jaykmody.com/blog/gpt-from-scratch/
    presentations:
      - name: XLNet
        url: https://arxiv.org/abs/1906.08237
      - name: UL2
        url: https://arxiv.org/pdf/2205.05131
      - name: OPT
        url: https://arxiv.org/abs/2205.01068
      - name: PaLM
        url: https://arxiv.org/abs/2204.02311
      - name: LLaMA
        url: https://arxiv.org/abs/2302.13971
      - name: Survey of Pre-trained LMs
        url: https://arxiv.org/pdf/2302.09419
    #homeworks:
#      - name: Huggingface Sentence Classification (Kaggle)
    #  - name: TBD
#        url: https://www.kaggle.com/competitions/sentence-classification-competition/overview
  - date: Nov.<br>10
    topics:
      - name: Large Language Models II - Prompt Tuning
#        url: https://www.dropbox.com/s/gcd1bu7bxd5gigm/Week8-DL4NLP-part3.pptx?dl=0
    readings:
      - name: Chain-of-Thought
        url: https://arxiv.org/pdf/2201.11903
      - name: Prompt Engineering (Blog by Lilian Weng)
        url: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
    presentations:
      - name: Prefix-Tuning
        url: https://arxiv.org/abs/2101.00190
      - name: Promtp Tuning
        url: https://arxiv.org/abs/2104.08691
      - name: LoRA
        url: https://arxiv.org/abs/2106.09685
      - name: Instruction Tuning
        url: https://arxiv.org/html/2308.10792v5
      - name: InstructGPT
        url: https://arxiv.org/abs/2203.02155
      - name: Automatic Prompt Engineer
        url: https://arxiv.org/abs/2211.01910
  - date: Nov.<br>17
    topics:
      - name: Generative Models
#        url: https://www.dropbox.com/s/nf4ohrqjqg7rb66/Week10-Graph-part2.pdf?dl=0
    readings:
      - name: GAN
        url: https://arxiv.org/abs/1406.2661
      - name: VAE
        url: https://arxiv.org/abs/1312.6114
      - name: Evidence Lower Bound ELBO — What & Why (Blog)
        url: https://yunfanj.com/blog/2021/01/11/ELBO.html
      - name: Diffusion Probabilistic Model
        url: https://arxiv.org/abs/2006.11239
    presentations:
      - name: beta-VAE
        url: https://openreview.net/pdf?id=Sy2fzU9gl
      - name: CycleGAN
        url: https://arxiv.org/pdf/1703.10593
      - name: Latent Diffusion
        url: https://arxiv.org/abs/2112.10752
      - name: Stable diffusion
        url: https://arxiv.org/abs/2112.10752
  #- date: Nov.<br>24
  #  topics:
  #    - name: Diffusion, text-to-image generation
# #       url: https://www.dropbox.com/s/nf4ohrqjqg7rb66/Week10-Graph-part2.pdf?dl=0
  #  readings:
  #    - name: CLIP
# #       url: https://arxiv.org/abs/2103.00020
  #    - name: Sora
# #       url: https://arxiv.org/abs/2201.00123
  - date: Nov.<br>24
    topics:
      - name: Graph Representation Learning
#        url: https://www.dropbox.com/s/3e09x5i9wyn8q3c/Week9-Graph-part1.pdf?dl=0
    readings:
      - name: GCN
        url: https://arxiv.org/pdf/1609.02907
#      - name: Bayesian Personalized Ranking
#        url: https://arxiv.org/abs/1205.2618
#      - name: Factorization Machines
#        url: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
      - name: Graph Neural Networks Implementation Tutorial 
        url: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
    presentations:
      - name: DeepWalk
        url: https://arxiv.org/pdf/1403.6652
      - name: LINE
        url: https://arxiv.org/pdf/1503.03578
      - name: GIN
        url: https://arxiv.org/abs/1810.00826
      - name: Open Graph Benchmark
        url: https://arxiv.org/abs/2005.00687
  - date: Dec.<br>1
    topics:
      - name: Poster Session
#   

---

# Schedule for In-class Presentations
<table>
  <tr>
    <th>Date</th>
    <th>Topic</th>
    <th style="width: 45%;">Suggested Readings</th>
    <th>Reference</th>
    <th>Homework</th>
  </tr>
  {% for week in page.schedule %}
    <tr>
      <td>{{ week.date }}</td>
      <td>
      {% for topic in week.topics %}
        {% include href item=topic %}<br>
      {% endfor %}
      </td>
      <td style="width: 45%;">
      {% for reading in week.readings %}
        {% include href item=reading %}<br>
      {% endfor %}
      </td>
      <td>
      {% for presentation in week.presentations %}
        {% include href item=presentation %}<br>
      {% endfor %}
      </td>
      <td>
      {% for homework in week.homeworks %}
        {% include href item=homework %}<br>
      {% endfor %}
      </td>
    </tr>
  {% endfor %}
</table>
