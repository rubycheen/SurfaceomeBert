# SurfaceomeBert

Surfaceome are considered as a signaling gateway to the cellular microenvironment, playing an important role in biomedicine. However, surfaceome discovery through traditional wet-lab research is expensive and inefficient. Thus, we applied a deep-learning approach utilizing Bert achetiecture to develop the accurate surfaceome predictor and used it to define the human protein on Uniprot with manual annotated (Swiss-prot) of 20,386 proteins. The positive dataset is mainly from CSPA highly-confident surfaceome and the positive dataset using by “The in silico human surfaceome”. And we chose the Bert pre-trained on protein sequences and replace the classic second pre-trained task “Next Sentence Prediction” with “Go Annotation Prediction” from “ProteinBERT: A universal deep-learning model of protein sequence and function”. Then we fine-tuned model on our surfaceome dataset and build up a web-server to accelerate the process of surfaceome discovery. 

## Website
AI4Surfaceome (web-server) is freely accessible at https://axp.iis.sinica.edu.tw/AI4Surfaceome/?fbclid=IwAR25sWVl8IpaN7ZEYtAadnZa8Ou00HLFRwcQ8jFADHHXIQjVLbhIYo5toh8


## Quick Start
Here we give a quick demo and command usage of our SurfaceomeBert.

### Environment Setup / Dependency Installation
SurfaceomeBert requires Python 3.
Install the Python packages required by SurfaceomeBert:

```
pip3 install -r requirements.txt
```

### Download Fine-tuned Model
You can get fine-tuned model from https://reurl.cc/XlVxe7 and just put the directory under root.
The path should be like `SurfaceomeBert/default`.


### Predict Your Own Proteins

For quick demo our model, run the command below

-f : input amino acids data in FASTA format

-m : SurfaceomeBert model using tensorflow framework

-o : output prediction result in CSV

```
python3 prediction.py -i ./test/example.fasta -m ./default/checkpoint -o ./test/example_output.csv
```


### Example Data

The default input of this demo is 10 human proteins (`test/example.fasta`) in FASTA format.

The prediction result (`test/example_output.csv`) below shows prediction scores and whether the protein is a surfaceome in table.

<img width="331" alt="截圖 2021-09-30 下午7 38 14" src="https://user-images.githubusercontent.com/56534481/135448372-bf8db363-2591-44f4-963d-07869504a4f9.png">


### Training Dataset
All protein sequences are from UniPort(https://www.uniprot.org/)

### Pre-trained Model
Pre-trained model is from https://github.com/nadavbra/protein_bert

### Ipynb Tutorial
If you like to continually fine-tuned this model by your amino acid sequences or use model in your code, please refer to `demo.ipynb` for implementation details of the project.

### Citation
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. 
ProteinBERT: A universal deep-learning model of protein sequence and function. 
Bioinformatics (2022). https://doi.org/10.1093/bioinformatics/btac020

```
@article{10.1093/bioinformatics/btac020,
    author = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
    title = "{ProteinBERT: a universal deep-learning model of protein sequence and function}",
    journal = {Bioinformatics},
    volume = {38},
    number = {8},
    pages = {2102-2110},
    year = {2022},
    month = {02},
    abstract = "{Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biological sequences. However, existing models and pretraining methods are designed and optimized for text analysis. We introduce ProteinBERT, a deep language model specifically designed for proteins. Our pretraining scheme combines language modeling with a novel task of Gene Ontology (GO) annotation prediction. We introduce novel architectural elements that make the model highly efficient and flexible to long sequences. The architecture of ProteinBERT consists of both local and global representations, allowing end-to-end processing of these types of inputs and outputs. ProteinBERT obtains near state-of-the-art performance, and sometimes exceeds it, on multiple benchmarks covering diverse protein properties (including protein structure, post-translational modifications and biophysical attributes), despite using a far smaller and faster model than competing deep-learning methods. Overall, ProteinBERT provides an efficient framework for rapidly training protein predictors, even with limited labeled data.Code and pretrained model weights are available at https://github.com/nadavbra/protein\_bert.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac020},
    url = {https://doi.org/10.1093/bioinformatics/btac020},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/8/2102/45474534/btac020.pdf},
}
```
