---
title: "Semantic Operators: A Declarative Model for Rich, AI-based Analytics Over Text Data"
summary: "LOTUS uses novel 'semantic operators' to enable declarative, AI-powered analytics over text data, achieving up to 400x speedups and exceeding state-of-the-art accuracy."
categories: ["AI Generated", ]
tags: ["Natural Language Processing", "Large Language Models", "🏢 Stanford University",]
showSummary: true
date: 2024-07-16
draft: false
---

<br>

{{< keywordList >}}
{{< keyword icon="fingerprint" >}} 2407.11418 {{< /keyword >}}
{{< keyword icon="writer" >}} Liana Patel et el. {{< /keyword >}}
 
{{< /keywordList >}}

{{< button href="https://arxiv.org/abs/2407.11418" target="_self" >}}
↗ arXiv
{{< /button >}}
{{< button href="https://huggingface.co/papers/2407.11418" target="_self" >}}
↗ Hugging Face
{{< /button >}}
{{< button href="https://paperswithcode.com/paper/lotus-enabling-semantic-queries-with-llms" target="_self" >}}
↗ Papers with Code
{{< /button >}}




### TL;DR


{{< lead >}}

Current systems lack high-level abstractions for performing bulk semantic queries on large text datasets.  This limits the potential of language models for advanced analytics.  The difficulty lies in expressing complex semantic operations and optimizing their execution efficiently.  Existing solutions either provide limited row-wise operations or lack the scalability and accuracy needed for bulk processing.

The paper proposes "semantic operators", a declarative programming interface that extends the relational model with AI-based operations. This allows users to compose AI operations (e.g., filtering, sorting, joining) in a flexible and intuitive manner.  The authors also implement several novel query optimizations in LOTUS, an open-source query engine, that leverage the declarative nature of the operators to significantly improve the performance (up to 400x) of various operations.  These optimizations are accompanied by statistical accuracy guarantees.  The work demonstrates LOTUS' effectiveness across several AI applications including fact-checking and extreme multi-label classification.

{{< /lead >}}


#### Key Takeaways

{{< alert "star" >}}
Semantic operators provide a high-level, declarative interface for complex AI-based analytics on text data.
{{< /alert >}}

{{< alert "star" >}}
LOTUS system significantly accelerates semantic queries (filtering, joining, ranking) by up to 400x while maintaining accuracy.
{{< /alert >}}

{{< alert "star" >}}
The approach is demonstrated on real-world applications, achieving state-of-the-art results in multiple tasks.
{{< /alert >}}

#### Why does it matter?
This paper is crucial for researchers working with large language models and data analytics.  It **introduces a novel declarative programming interface** and **efficient query optimization techniques** that significantly improve the speed and accuracy of semantic queries. This is highly relevant to current trends in AI-based analytics and opens exciting new avenues for research and development of large-scale AI applications.

------
#### Visual Insights



![](https://arxiv.org/html/2407.11418/x1.png)

> 🔼 This figure compares the execution time and accuracy of three different fact-checking implementations. The first is FacTool, a state-of-the-art model. The second is a LOTUS program, and the third is the same LOTUS program but with optimizations enabled. The optimizations significantly improve execution time while maintaining accuracy.  Section 5 of the paper provides details on the methodology used in the comparison.
> <details>
> <summary>read the caption</summary>
> Figure 1. Execution time (y-axis) and accuracy, shown in parentheses, for 3 fact-checking implementations: (i) FacTool (Chern et al., 2023), a recent state-of-the-art research work, (ii) a short LOTUS program, and (iii) the same LOTUS program implemented with our declarative optimizations and accuracy guarantees. Section 5 provides our full methodology.
> </details>





{{< table-caption >}}
| Operator | Description |
|---|---| 
| \mathit{sem_filter}(l\textit{: }X\rightarrow\mathit{Bool}) | Returns the tuples in a table that pass the provided langex predicate. |
| \mathit{sem_join}(t\textit{: }T{\textit{,   }{l}}\textit{: }(X,Y)\rightarrow \mathit{Bool}) | Joins a table against a second table  $t$ by keeping all pairs of tuples that pass the provided langex predicate. |
| \mathit{sem_agg}(l\textit{: }L[X]\rightarrow\mathit{X}) | Performs an aggregation over the input tuples according to the langex, which specifies a commutative, associative aggregation function over a list of tuples. |
| \mathit{sem_topk}(l\textit{: }L[X]\rightarrow L[X]{\textit{,   }{k}}\textit{: }int) | Ranks each tuple and returns the $k$ best according to the langex, which specifies a ranking function that sorts a list of tuples. |
| \mathit{sem_group_by}(l\textit{: }X\rightarrow Y{\textit{,   }{C}}\textit{: }int) | Groups the tuples into $C$ categories based on the langex, which specifies a grouping criteria. |
| \mathit{sem_map}(l\textit{: }X\rightarrow Y) | Performs a projection, returning a new column, according to the provided langex. |{{< /table-caption >}}

> 🔼 This table summarizes the key semantic operators used in the LOTUS system.  It details the function of each operator, including the input types (relation T, tuple types X and Y, list of elements L[X], and attribute type A), and the parameterized natural language expression (langex, l) used to define the operation. The langex serves as a flexible way to specify predicates, aggregations, comparisons, or projections, depending on the operator. The table clarifies how these operators extend the relational model with AI-based capabilities for querying text data.
> <details>
> <summary>read the caption</summary>
> Table 1. Summary of Key Semantic Operators. T𝑇Titalic_T denotes a relation, X𝑋Xitalic_X and Y𝑌Yitalic_Y denote arbitrary tuple types, L⁢[X]𝐿delimited-[]𝑋L[X]italic_L [ italic_X ] denotes a list of elements with type X𝑋Xitalic_X, and A𝐴Aitalic_A denotes the type of a particular column or attribute. l𝑙litalic_l denotes a parameterized natural language expression (“langex” for short), which takes tuples as input and performs a function such as a predicate, an aggregation, a comparator, or a projection, depending on the operator’s signature.
> </details>





### In-depth insights


#### Semantic Querying
Semantic querying aims to bridge the gap between human language and database interaction.  It moves beyond keyword-based searches by understanding the **meaning and context** of user queries. This allows for more natural and intuitive interactions with data, enabling users to ask complex questions without needing specialized syntax. **Natural language processing (NLP)** techniques are crucial for semantic querying, as they enable the system to interpret the intent behind the words used.  A key challenge lies in the ambiguity inherent in human language; systems must resolve this ambiguity to produce accurate results.  **Knowledge representation and reasoning** play a vital role; the system needs a way to represent the data's semantics and infer relationships to answer complex queries effectively. The power of semantic querying lies in its capacity to unlock the knowledge hidden within large datasets, making it accessible to a wider range of users.  However, the need for sophisticated NLP and knowledge representation makes building effective semantic querying systems a complex undertaking.  There is also the need to consider **scalability** to handle ever-growing datasets.

#### LOTUS Framework
The LOTUS framework, as described in the research paper, presents a novel approach to declarative AI-based analytics over text data.  **Its core innovation lies in introducing semantic operators**, extending the relational model with AI-powered operations for bulk semantic queries.  This allows users to express complex analytical tasks using natural language, abstracting away low-level implementation details. The system's **declarative nature enables efficient query optimization and planning**, similar to traditional relational database systems, but for semantic tasks.  **LOTUS leverages multiple optimizations**, including novel techniques for semantic filtering, joins, and aggregation.  These optimizations greatly enhance the speed and efficiency of processing large-scale datasets while maintaining accuracy. The open-source implementation makes the framework accessible for broader usage and experimentation. The **expressiveness of the semantic operators is highlighted by achieving state-of-the-art results** on various applications including fact-checking, multi-label classification, and search, which demonstrates its practical utility and impact.  This framework is a valuable contribution in bridging the gap between powerful language models and efficient data analytics.

#### Optimizer Details
An 'Optimizer Details' section in a research paper on semantic operators would delve into the specifics of how the system efficiently executes queries using AI-based operations. This would involve a detailed explanation of the optimization strategies employed for various semantic operators (**filter, join, top-k, group-by**).  The discussion would likely cover the trade-offs between accuracy and efficiency, especially highlighting any approximation techniques used and their associated guarantees.  **Lossless optimizations** for specific operators would be showcased, contrasting them with approximation algorithms offering statistical accuracy guarantees.  The discussion would also address how the optimizer manages the computational cost of interacting with large language models (LLMs), possibly through techniques like **batch processing, model cascading, or the use of proxy models** for faster computations. Furthermore, it would delve into the **implementation details** of data structures and algorithms to provide a more comprehensive understanding of the performance characteristics of the system.  Finally, the section would likely include empirical evidence showcasing the performance improvements achieved by the optimizer on real-world tasks.

#### Empirical Studies
An Empirical Studies section in a research paper would rigorously evaluate the proposed methods.  It would likely present results from multiple experiments, comparing the performance of the new approach against existing state-of-the-art techniques.  **Metrics such as accuracy, efficiency (execution time), and scalability would be crucial**.  The choice of datasets used for evaluation would also be a key factor; a well-designed study would employ diverse datasets to demonstrate the generalizability of the results.  A strong 'Empirical Studies' section would also involve **robust statistical analysis** to assess the significance of any observed performance differences.  **Detailed descriptions of experimental setups**, including parameter choices and any pre-processing steps, would be essential to ensure reproducibility.  The discussion of results would focus on identifying trends, explaining unexpected findings, and highlighting the limitations of the approach.  Overall, a compelling 'Empirical Studies' section would provide strong evidence supporting the claims made in the paper, **strengthening the credibility and impact of the research**.

#### Future Work
Future research directions stemming from this work on semantic operators are multifaceted.  **Extending the range of supported semantic operators** is crucial; incorporating temporal reasoning, advanced statistical functions (beyond simple aggregations), and geospatial analysis would greatly broaden the system's capabilities.  **Improving the efficiency of complex queries** is another key area; exploring novel query optimization strategies beyond the current cascade approach, such as exploiting query rewriting techniques and incorporating learned query plans, would enhance performance.  **Addressing the limitations of current LLMs** is paramount; the dependency on LLMs' context windows necessitates strategies for handling extremely large datasets or very complex queries.  Investigating techniques such as hybrid approaches (combining LLMs with other methods) or efficient indexing methods specifically designed for semantic queries may prove beneficial.  Finally, **enhancing the user experience** through an improved interface and tools for visualizing query execution is important.  The ultimate goal is a system that is not only powerful and efficient but also intuitive and easily accessible to a wider range of users, allowing broader adoption and pushing the boundaries of semantic analytics.


### More visual insights

<details>
<summary>More on figures
</summary>


![](https://arxiv.org/html/2407.11418/x2.png)

> 🔼 The figure showcases a concise LOTUS program (Python code) that leverages semantic operators to efficiently process and summarize research papers.  The program begins by using `sem_search` to retrieve papers relevant to specified research interests.  Next, `sem_filter` refines the results, selecting only papers claiming to outperform a given baseline. Finally, `sem_agg` generates a summary of the remaining papers' abstracts. This example highlights LOTUS's declarative, composable nature, enabling users to perform complex, AI-powered analyses using a high-level, natural-language-like interface.
> <details>
> <summary>read the caption</summary>
> Figure 2. Example LOTUS program using semantic operators to return a summary of relevant papers. The function takes a description of the user’s research interests. The program searches over papers, then filters based on whether the paper outperforms the baseline, and finally constructs a summary.
> </details>



![](https://arxiv.org/html/2407.11418/x3.png)

> 🔼 The figure demonstrates the difference between two types of joins: sem_join and sem_sim_join.  The sem_join performs a join based on a user-specified natural language predicate that must be true for two rows to be joined, whereas sem_sim_join performs a join based on semantic similarity.  The example shows that sem_sim_join can retrieve multiple matches (here, K=10) from the right table for each row in the left table, based on similarity scores, whereas sem_join only retrieves rows that exactly satisfy the predicate.
> <details>
> <summary>read the caption</summary>
> Figure 3. Example usage of sem_join and sem_sim_join.
> </details>



</details>




<details>
<summary>More on tables
</summary>


{{< table-caption >}}
| Method | Accuracy | Execution Time (s), batched | Execution Time (s), no batching | LoC |
|---|---|---|---|---|
| FacTool | 80.9 | N/A | 5396.11 | ~ 750 |
| LOTUS-Factool | 89.9 | 688.90 | 4,454.24 | ~ 50 |
| LOTUS-fact-filter | 91.2 | 329.1 | 988.95 | ~ 50 |
| LOTUS-fact-filter (opt.)* | 91.0 | 189.88 | 776.37 | ~ 50 |
| LOTUS-fact-join | 84.5 | 11,951.35 | 60,364.39 | ~ 50 |{{< /table-caption >}}
> 🔼 This table presents the results of fact-checking experiments conducted on the FEVER dataset.  It compares the accuracy, execution time (both with and without batching), and lines of code of different methods: FacTool (a state-of-the-art baseline), and three LOTUS programs with varying implementations (re-implementation of FacTool, filter-based, and join-based approaches). The table highlights the performance and efficiency of LOTUS in solving the fact-checking task, demonstrating that LOTUS requires significantly fewer lines of code while achieving comparable or even improved accuracy with shorter execution times compared to the baseline.
> <details>
> <summary>read the caption</summary>
> Table 2. Fact-checking Results on the Fever Dataset.
> </details>

{{< table-caption >}}
| Method | RP@5 | RP@10 | Execution Time (s) | # LM Calls |
|---|---|---|---|---|
| Sem-sim-join | 0.106 | 0.120 | 2.91 | 0.00 |
| LOTUS Sem-join** | 0.244 | 0.261 | 5,891.6 | 15,107 |
| nested-loop pattern | N/A | N/A | 2,144,560* | 6,092,500 |
| map-sim-filter pattern** | 0.244 | 0.261 | 5,891.6 | 15,107 |
| sim-filter pattern** | 0.154 | 0.191 | 24,206.8 | 63,724 |{{< /table-caption >}}
> 🔼 This table presents the results of extreme multi-label classification experiments conducted on the Biodex dataset using the Llama-70b language model.  It compares the performance of several methods, including a simple semantic similarity join and different variants of the LOTUS semantic join. The metrics used for evaluation include accuracy, rank-precision@5 (RP@5), and rank-precision@10 (RP@10).  Execution time and the number of large language model calls are also reported, offering insight into computational costs and efficiency.
> <details>
> <summary>read the caption</summary>
> Table 3. Extreme Multi-label Classification Results on Biodex Dataset with Llama-70b
> </details>

{{< table-caption >}}
| Method | nDCG@10 | ET (s) |
|---|---|---|
| Search | 0.712 | 0.009 |
| Search + Reranker | 0.741 | 2.64 |
| LOTUS Top-k - Llama-70B | 0.775 | 33.6 |
| LOTUS Top-k - GPT-4o | 0.800 | 11.2 |{{< /table-caption >}}
> 🔼 This table presents the results of the SciFact ranking task, comparing different methods.  It shows the nDCG@10 (normalized Discounted Cumulative Gain at rank 10) scores, a metric measuring ranking quality, along with the execution time (ET) for each approach.  The methods compared include simple semantic search, search with a reranker, and the LOTUS system using two different language models (Llama-70B and GPT-40).  Higher nDCG@10 indicates better ranking performance, while lower ET suggests faster query execution.
> <details>
> <summary>read the caption</summary>
> Table 4. Ranking Results on SciFact
> </details>

{{< table-caption >}}
| Method | CIFAR nDCG@10 | CIFAR ET (s) | HellaSwag nDCG@10 | HellaSwag ET (s) |
|---|---|---|---|---|
| Search | 0.252 | 0.008 | 0.119 | 0.008 |
| Search + Reranker | 0.001 | 2.57 | 0.461 | 2.36 |
| LOTUS Top-k - Llama 70B | 0.710 | 39.6 | 0.975 | 63.6 |{{< /table-caption >}}
> 🔼 This table presents the results of a ranking task performed on two newly created benchmarks: CIFAR-bench and HellaSwag-bench.  These benchmarks evaluate the ability of different methods to rank scientific papers based on their reported accuracy for the CIFAR-10 and HellaSwag datasets. The table compares the performance of three methods: a semantic search, a semantic search with a re-ranker, and the LOTUS system's semantic top-k approach using the Llama 70B language model. For each method, the table shows the nDCG@10 score (a measure of ranking quality) and the execution time in seconds.
> <details>
> <summary>read the caption</summary>
> Table 5. Ranking Results on CIFAR- &amp; HellaSwag-bench
> </details>

{{< table-caption >}}
| Method | Scifact nDCG@10 | Scifact ET (s) | Scifact # LM Calls | CIFAR nDCG@10 | CIFAR ET (s) | CIFAR # LM Calls | HellaSwag nDCG@10 | HellaSwag ET (s) | HellaSwag # LM Calls |
|---|---|---|---|---|---|---|---|---|---| 
| Quadratic Sort | 0.836 | 712 | 4950 | 0.868 | 634 | 4950 | 0.966 | 1,803 | 19,900 |
| Heap Top-k | 0.776 | 65.0 | 216 | 0.832 | 99.6 | 350 | 0.907 | 98.9 | 415.2 |
| QuickSelect Top-k | 0.776 | 42.4 | 285 | 0.746 | 41.3 | 303.95 | 0.909 | 59.1 | 620.95 |
| QuickSelect Top-k + Semantic Index | 0.775 | 33.6 | 229 | 0.710 | 39.6 | 307.7 | 0.975 | 63.6 | 672.7 |{{< /table-caption >}}
> 🔼 This table compares the performance of four different algorithms used for semantic top-k ranking, all using the Llama-70B language model.  It shows the nDCG@10 score (a measure of ranking quality), execution time in seconds, and the number of Language Model calls required for each algorithm across three different datasets: SciFact, CIFAR-bench, and HellaSwag-bench. The algorithms compared are Quadratic Sort, Heap Top-k, QuickSelect Top-k, and a QuickSelect Top-k algorithm optimized with a semantic index. The datasets vary in complexity, with SciFact focusing on relevance ranking and the other two using more complex ranking criteria based on reported accuracy. This allows for a comprehensive comparison of the algorithms under various conditions.
> <details>
> <summary>read the caption</summary>
> Table 6. Comparison of Ranking Results for Different Semantic Top-k Algorithms using Llama-70B
> </details>

{{< table-caption >}}
| μ | Description |
|---|---| 
| μ₁ | Advancements in Recommender Systems and Multimodal Data Integration |
| μ₂ | Advancements in Generative Information Retrieval Systems |
| μ₃ | Advancements in Large Language Models for Various Applications |
| μ₄ | Advancements in AI Security and Malware Detection Techniques |
| μ₅ | Advancements in Robotic Navigation and Manipulation Techniques |{{< /table-caption >}}
> 🔼 This table presents the five group labels automatically discovered by the LOTUS system when applied to a dataset of 647 arXiv papers.  Each label summarizes a common theme or topic found within a cluster of papers. The goal of this unsupervised grouping task is to demonstrate the capabilities of the semantic group-by operator in LOTUS and the quality of its automatic label generation.  The effectiveness of these automatically generated labels is further evaluated in a later section of the paper by comparing the accuracy of a classifier that uses these labels against other baselines.
> <details>
> <summary>read the caption</summary>
> Table 7. Discovered Group Labels Over ArXiv Papers
> </details>

</details>




### Full paper

{{< gallery >}}
<img src="paper_images/1.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/2.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/3.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/4.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/5.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/6.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/7.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/8.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/9.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/10.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/11.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/12.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/13.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
<img src="paper_images/14.png" class="grid-w50 md:grid-w33 xl:grid-w25" />
{{< /gallery >}}
