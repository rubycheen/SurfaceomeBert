# SurfaceomeBert

Surfaceome are considered as a signaling gateway to the cellular microenvironment, playing an important role in biomedicine. However, surfaceome discovery through traditional wet-lab research is expensive and inefficient. Thus, we applied a deep-learning approach utilizing Bert achetiecture to develop the accurate surfaceome predictor and used it to define the human protein on Uniprot with manual annotated (Swiss-prot) of 20,386 proteins. The positive dataset is mainly from CSPA highly-confident surfaceome and the positive dataset using by “The in silico human surfaceome”. And we chose the Bert pre-trained on protein sequences and replace the classic second pre-trained task “Next Sentence Prediction” with “Go Annotation Prediction” from “ProteinBERT: A universal deep-learning model of protein sequence and function”. Then we fine-tuned model on our surfaceome dataset and build up a web-server to accelerate the process of surfaceome discovery. 

### Website
TBA

### Dataset
All protein sequences are from UniPort(https://www.uniprot.org/)

### Pre-trained model
Pre-trained model is from https://github.com/nadavbra/protein_bert

### Predict your own proteins
If you like to use this model to predict your amino acid sequences, please refer to demo.ipynb for implementation details of the project.

### Reference
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. ProteinBERT: A universal deep-learning model of protein sequence and function. bioRxiv (2021). https://doi.org/10.1101/2021.05.24.445464
