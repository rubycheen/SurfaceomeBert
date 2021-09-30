# SurfaceomeBert

Surfaceome are considered as a signaling gateway to the cellular microenvironment, playing an important role in biomedicine. However, surfaceome discovery through traditional wet-lab research is expensive and inefficient. Thus, we applied a deep-learning approach utilizing Bert achetiecture to develop the accurate surfaceome predictor and used it to define the human protein on Uniprot with manual annotated (Swiss-prot) of 20,386 proteins. The positive dataset is mainly from CSPA highly-confident surfaceome and the positive dataset using by “The in silico human surfaceome”. And we chose the Bert pre-trained on protein sequences and replace the classic second pre-trained task “Next Sentence Prediction” with “Go Annotation Prediction” from “ProteinBERT: A universal deep-learning model of protein sequence and function”. Then we fine-tuned model on our surfaceome dataset and build up a web-server to accelerate the process of surfaceome discovery. 

## Website
AI4Surfaceome (web-server) is freely accessible at TBA.


## Quick Start
Here we give a quick demo and command usage of our SurfaceomeBert.

### Environment Setup (Dependency Installation)
SurfaceomeBert requires Python 3.
Install the Python packages required by SurfaceomeBert:

```
pip3 install -r requirements.txt
```

### Predict Your Own Proteins

For quick demo our model, run the command below

-f : input amino acids data in FASTA format

-m : SurfaceomeBert model using tensorflow framework

-o : output prediction result in CSV

```
python3 prediction.py -i ./test/example.fasta -m ./default/checkpoint -o ./test/example_output.csv
```


The default input of this demo is 10 human proteins (`test/example.fasta`) in FASTA format.

The prediction result (`test/example_output.csv`) below shows prediction scores and whether the protein is a surfaceome in table.

<img width="331" alt="截圖 2021-09-30 下午7 38 14" src="https://user-images.githubusercontent.com/56534481/135448372-bf8db363-2591-44f4-963d-07869504a4f9.png">


### SurfaceomeBert Deep Neural Network Model Architecture
TBA

### Dataset
All protein sequences are from UniPort(https://www.uniprot.org/)

### Pre-trained Model
Pre-trained model is from https://github.com/nadavbra/protein_bert

### Ipynb Tutorial
If you like to use this model to predict your amino acid sequences, please refer to demo.ipynb for implementation details of the project.

### Reference
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. ProteinBERT: A universal deep-learning model of protein sequence and function. bioRxiv (2021). https://doi.org/10.1101/2021.05.24.445464
