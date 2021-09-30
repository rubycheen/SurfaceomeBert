from utils import get_prediction, load_fituned_model, read_fasta
from proteinbert import load_pretrained_model
import pandas as pd
import argparse


def main(input_fasta_name, model_path, output_csv_name):

    ############################################
    ##               load data                ##
    ############################################

    MAX_LEN=512

    example_10 = read_fasta(input_fasta_name)

    sentences = []
    for p in example_10.values():
        sentences.append(p[:MAX_LEN-2])


    ############################################
    ##       load model and get prediction    ##
    ############################################

    model = load_fituned_model(model_path=model_path, seq_len=512)

    ypred = get_prediction(sentences, model)

    df = pd.DataFrame({'Protein': list(example_10.keys()),
                       'Score': ypred.flatten(),
                       'Prediction results': ypred.flatten() >= 0.5})
    df.to_csv(output_csv_name,index=False)

    
#args
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Surfaceome predictor')
    parser.add_argument('-f','--fasta_name', help='input fasta name', default='./test/example_10.fasta')
    parser.add_argument('-m','--model_path', help='output csv name', default='./default/checkpoint')
    parser.add_argument('-o','--output_csv', help='output csv name', default='./test/example_output.csv')
    args = parser.parse_args()
    input_fasta_name = args.fasta_name
    model_path =  args.model_path
    output_csv_name =  args.output_csv
    print(input_fasta_name,model_path,output_csv_name)
    main(input_fasta_name, model_path, output_csv_name)