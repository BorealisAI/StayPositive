## WordNet:
---

### Source Description

WordNet® is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations. For more information on the data source, kindly refer to this link (https://wordnet.princeton.edu).

**Source Reference**:
```
@article{WordNetMiller,
  title={WordNet: A Lexical Database for English.},
  author={George A. Miller},
  journal={Communications of the ACM},
  volume={38},
  year={1995}
}

@article{boschee2015icews,
  title={WordNet: An Electronic Lexical Database},
  author={Christiane Fellbaum},
  journal={Cambridge, MA: MIT Press},
  year={1998}
}
```

### Dataset Description

The two datasets used in this work are subsets of the data present in the WordNet repository: 

* WN18RR: This is a subset of the Wordnet dataset (WN18) with reduced number of relations that does not include test set leakage through inverse relations. All sets, train, validation, and test, include only positive examples.

* WN11: This is also a subset of the Wordnet (WN18) with reduced number of entities and relations. However, it contains negative examples in both validation and test sets such that half of the examples are negative.

* WN18AM: This dataset is a modified version of WN18RR. Due to the previous filtering for leakage problem, some of the entities/relations in the test set of WN18RR do not appear in the training set of this dataset. 
This issue is fixed in WN18AM.

**Dataset reference:**
```
@inproceedings{dettmers2018convolutional,
  title={Convolutional 2d knowledge graph embeddings},
  author={Dettmers, Tim and Minervini, Pasquale and Stenetorp, Pontus and Riedel, Sebastian},
  booktitle={AAAI},
  year={2018}
}

@inproceedings{socher2013reasoning,
  title={Reasoning with neural tensor networks for knowledge base completion},
  author={Socher, Richard and Chen, Danqi and Manning, Christopher D and Ng, Andrew},
  booktitle={AAAI},
  pages={926--934},
  year={2013}
}
```

## Freebase:
---

### Source Description 

Freebase is an online collection of structured data harvested from many sources, including individual, user-submitted wiki contributions, that is currently deprecated. Freebase aimed to create a global resource that allowed people (and machines) to access common information more effectively. For more information on the data source, kindly refer to this link (https://developers.google.com/freebase).

**Source Reference:**
```
@misc{freebase:datadumps,
  title = {Freebase Data Dumps}
  author = {Google},
  howpublished = {https://developers.google.com/freebase/data},
  }

@inproceedings{bollacker2008freebase,
  title={Freebase: a collaboratively created graph database for structuring human knowledge},
  author={Bollacker, Kurt and Evans, Colin and Paritosh, Praveen and Sturge, Tim and Taylor, Jamie},
  booktitle={ACM SIGMOD},
  year={2008},
  organization={AcM}
}
```

### Dataset Description 

The two datasets used in this work are subsets of the data present in the Freebase repository: 

* FB15k-237: This is a subset of the freebase dataset (FB15K) with reduced number of relations and entities. It is a cleaner version of FB15K where the test set leakage through inverse relations is removed. Its data sets, train, validation, and test sets, only contain positive examples.

* FB13: This is a subset of the freebase dataset (FB15K) with reduced number of relations but much larger number of entities. Its validation and test sets include both positive and negative examples (half negative).

* FB15k-237AM: This dataset is a modified version of FB15k-237. Due to the previous filtering for leakage problem, some of the entities/relations in the test set of FB15k-237 do not appear in the training set of this dataset.
This issue is fixed in FB15k-237AM, which follows the Freebase license that is Create Commons Attribution 
(a.k.a CC-BY): https://creativecommons.org/licenses/by/2.5/.

**Dataset reference:**
```
@inproceedings{toutanova2015observed,
  title={Observed versus latent features for knowledge base and text inference},
  author={Toutanova, Kristina and Chen, Danqi},
  booktitle={Continuous Vector Space Models and their Compositionality},
  pages={57--66},
  year={2015}
}

@inproceedings{socher2013reasoning,
  title={Reasoning with neural tensor networks for knowledge base completion},
  author={Socher, Richard and Chen, Danqi and Manning, Christopher D and Ng, Andrew},
  booktitle={AAAI},
  pages={926--934},
  year={2013}
}
```