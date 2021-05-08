# Explainable NLG Metrics

This project aims at *explaining* state-of-the-art NLG metrics, including
* Monolingual metrics, in particular [BertScore](https://openreview.net/forum?id=SkeHuCVFDr) and [SBERT](https://github.com/UKPLab/sentence-transformers); and
* Crosslingual metrics, in particular [XMoverScore](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation).
We provide explanations by *breaking down* the similarity score to show the *contribution* of each word. The computation was performed by using the [SHAP](https://github.com/slundberg/shap) method.  

![sts-example](docs/expl-sts-example.png)

In the example above, it shows that the contributions of words *Bordeaux* and *Luzon* are negative, suggesting that their appearance *harms* the similarity score.  

![xmover-example](docs/expl-xmover-example.png)
In the example above, the quanlity of translations are measured by XMoverScores, and the score breakdown suggests that word *dislikes* harms the score. 

More monolingual examples can be found at [here](sts_example.ipynb), and crosslingual examples can be found at [here](xmover_example.ipynb)


**Contact person**: [Yang Gao](https://sites.google.com/site/yanggaoalex/home)@Royal Holloway, Unversity of London. Don't hesitate to drop me an e-mail if something is broken or if you have any questions. 


## License
Apache License Version 2.0
