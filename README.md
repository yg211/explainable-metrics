# Explainable STS

This project provides a method that can 
* **measure the semantic similarity** between sentence pairs; and  
* **explain the similarity score**, by breaking down the score to the **contribution** of each word. 

![example-use-case](docs/expl-sts-example.png)

In the example above, it shows that the contributions of words *Bordeaux* and *Luzon* are negative, suggesting that their appearance *harms* the similarity score.  More examples can be found at [example](example.ipynb).

Contact person: [Yang Gao](https://sites.google.com/site/yanggaoalex/home)@Royal Holloway, Unversity of London. Don't hesitate to drop me an e-mail if something is broken or if you have any questions

## STS Model
Our semantic similarity measure model was developed based on BERT-large. It was pre-trained with SNLI and MLI, and fine-tuned on the [STSb dataset](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) and a dataset adapted from [HANS](https://github.com/tommccoy1/hans). Experiments show it performs better than the SOTA STS model, [SBERT](https://github.com/UKPLab/sentence-transformers), on the STSb benchmark. Our model is particularly better at rating sentences that *share many words but deliver different meanings*,  as the table below shows. 

| Sentence Pair | SBERT Score | Our Score | 
|---------------|-------------|-----------|
|'Charlton coach Guy Luzon had said on Monday', 'Charlton coach Bordeaux had said on Monday' | .672 | .534 |
|'Snow was predicted later in the weekend for Atlanta and areas even further south.', 'Snow wasn’t predicted later in the weekend for Atlanta and areas even further south.' | .683 | .416 |
|'Tom is his father', 'Tom is her dad' | .926 | .554 |
|'Her birthday was in July', 'Her birthday was before July' | .761 | .576 |


## Explanations
We provide explanations by *breaking down* the similarity score to show the *contribution* of each word. The computation was performed by using the [SHAP](https://github.com/slundberg/shap) method.  


## License
Apache License Version 2.0
