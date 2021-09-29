import pandas as pd
import numpy as np
from IPython.display import display

from tensorflow import keras

from sklearn.model_selection import train_test_split

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


def get_prediction(data, model, BATCH_SIZE=32, SEQ_LEN=512):
    
    pretrained_model_generator, input_encoder = load_pretrained_model()
    X = input_encoder.encode_X(data, seq_len=SEQ_LEN)

    prediciton = model.predict(X, batch_size = BATCH_SIZE)

    return prediciton


def read_fasta(fasta_fname):
    
    from Bio import SeqIO
    
    path = fasta_fname
    r = dict()
    for record in SeqIO.parse(path, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        r[idtag] = seq
    return r

def negtive_sampling(negtive, pos_len):
    
    import random
    
    neg_list = list(negtive.items())
    neg = random.sample(neg_list, pos_len)
    negtive = dict(neg)
    negtive_sentences = []
    for n in negtive.values():
        negtive_sentences.append(n[:510])
    negitive_labels = [0]*len(negtive_sentences)

    return negtive_sentences, negitive_labels

def save_fituned_model(model, path='./default'):
    
    import os
        
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_weights(path+'/checkpoint')
    print("Save model at",path)

def load_fituned_model(model_path='./default', seq_len=512):
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
    SEQ_LEN = 512
    
    pretrained_model_generator, input_encoder = load_pretrained_model()
    
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function =             get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)
    
    model = model_generator.create_model(SEQ_LEN)
    model.load_weights(model_path)
    return model
    
def validation(model, input_encoder, output_spec, seqs, raw_Y, start_seq_len = 512, start_batch_size = 32, increase_factor = 2):
        
    dataset = pd.DataFrame({'seq': seqs, 'raw_y': raw_Y})
        
    results = []
    results_names = []
    y_trues = []
    y_preds = []
    
    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len(dataset, start_seq_len = start_seq_len, start_batch_size = start_batch_size,increase_factor = increase_factor):

        X, y_true, sample_weights = encode_dataset(len_matching_dataset['seq'], len_matching_dataset['raw_y'], input_encoder, output_spec, seq_len = seq_len, needs_filtering = False)
        
        assert set(np.unique(sample_weights)) <= {0.0, 1.0}
        y_mask = (sample_weights == 1)
            
        y_pred = model.predict(X, batch_size = batch_size)
        
        y_true = y_true[y_mask].flatten()
        y_pred = y_pred[y_mask]
        
        if output_spec.output_type.is_categorical:
            y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        else:
            y_pred = y_pred.flatten()
        
        results.append(get_evaluation_results(y_true, y_pred, output_spec))
        results_names.append(seq_len)
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        
    y_true = np.concatenate(y_trues, axis = 0)
    y_pred = np.concatenate(y_preds, axis = 0)
    all_results, confusion_matrix = get_evaluation_results(y_true, y_pred, output_spec, return_confusion_matrix = True)
    results.append(all_results)
    results_names.append('All')
    
    results = pd.DataFrame(results, index = results_names)
    results.index.name = 'Model seq len'
    
    return results, confusion_matrix, model

def model_fituning(model_name, train_set, test_set, save_model=False, path='./default'):
    # A local (non-global) bianry output
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
    SEQ_LEN = 512

    # Loading the dataset
    train_set, valid_set = train_test_split(train_set, stratify = train_set['label'], test_size = 0.1, random_state = 42)
    print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')


    # Loading the pre-trained model and fine-tuning it on the loaded dataset
    pretrained_model_generator, input_encoder = load_pretrained_model()

    # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function =             get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-07, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ]
    
    finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], seq_len = SEQ_LEN, batch_size = 32, max_epochs_per_stage = 1, lr = 1e-05, begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-07, callbacks = training_callbacks)
    
    model = model_generator.create_model(seq_len = SEQ_LEN)
        
    # Evaluating the performance on the test-set
    results, confusion_matrix, model = validation(model, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], start_seq_len = SEQ_LEN, start_batch_size = 32)
    
    if save_model:
        save_fituned_model(model, path)

    print('Test-set performance:')
    display(results)

    print('Confusion matrix:')
    display(confusion_matrix)
    
    return model, results, confusion_matrix
    
def valid_report(title, results, confusion_matrixs):
    acc = []
    pre = []
    rec = []
    for i in range(len(confusion_matrixs)):
        tn = confusion_matrixs[i]['0']['0']
        fn = confusion_matrixs[i]['0']['1']
        fp = confusion_matrixs[i]['1']['0']
        tp = confusion_matrixs[i]['1']['1']
        Accuracy = (tp+tn)/(tp+fp+fn+tn)
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        print('Fold K',i,'Accuracy:',Accuracy,'Precision:',Precision,'Recall:',Recall)
        acc.append(Accuracy)
        pre.append(Precision)
        rec.append(Recall)

    print(title)
    print('AUC:',sum(results)/len(results))
    print('Average Accuracy:',sum(acc)/len(acc))
    print('Average Precision:',sum(pre)/len(pre))
    print('Average Recall:',sum(rec)/len(rec))

def cross_validation(sentences, labels, fix_training_seqs=None, fix_training_labels=None):
    k=10
    results = []
    confusion_matrixs = []

    for fold_index in range(k):
        train = []
        train_labels = []
        test = []
        test_labels = []

        for s in range(len(sentences)):
            if s%k==fold_index:
                test.append(sentences[s])
                test_labels.append(labels[s])
            else:
                train.append(sentences[s])
                train_labels.append(labels[s])
        if fix_training_seqs==None:
            train_set = pd.DataFrame(data={'label':train_labels,'seq':train})
        else:
            train_set = pd.DataFrame(data={'label':train_labels+fix_training_labels,'seq':train+fix_training_seqs})
        test_set = pd.DataFrame(data={'label':test_labels,'seq':test})
        r, c, model_generator = model('K'+str(fold_index), train_set, test_set)
        results.append(r)
        confusion_matrixs.append(c)
        
    return results, confusion_matrixs

def encode_Y(raw_Y, output_spec, seq_len = 512):
    
    if output_spec.output_type.is_seq:
        return encode_seq_Y(raw_Y, seq_len, output_spec.output_type.is_binary, output_spec.unique_labels)
    elif output_spec.output_type.is_categorical:
        return encode_categorical_Y(raw_Y, output_spec.unique_labels), np.ones(len(raw_Y))
    elif output_spec.output_type.is_numeric or output_spec.output_type.is_binary:
        return raw_Y.values.astype(float), np.ones(len(raw_Y))
    else:
        raise ValueError('Unexpected output type: %s' % output_spec.output_type)

def encode_dataset(seqs, raw_Y, input_encoder, output_spec, seq_len = 512, needs_filtering = True, dataset_name = 'Dataset', verbose = True):
    
    if needs_filtering:
        dataset = pd.DataFrame({'seq': seqs, 'raw_Y': raw_Y})
        dataset = filter_dataset_by_len(dataset, seq_len = seq_len, dataset_name = dataset_name, verbose = verbose)
        seqs = dataset['seq']
        raw_Y = dataset['raw_Y']
    
    X = input_encoder.encode_X(seqs, seq_len)
    Y, sample_weigths = encode_Y(raw_Y, output_spec, seq_len = seq_len)
    return X, Y, sample_weigths

def split_dataset_by_len(dataset, seq_col_name = 'seq', start_seq_len = 512, start_batch_size = 32, increase_factor = 2):
    
    seq_len = start_seq_len
    batch_size = start_batch_size
    ADDED_TOKENS_PER_SEQ = 2
    
    while len(dataset) > 0:
        max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
        len_mask = (dataset[seq_col_name].str.len() <= max_allowed_input_seq_len)
        len_matching_dataset = dataset[len_mask]
        yield len_matching_dataset, seq_len, batch_size
        dataset = dataset[~len_mask]
        seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)
        
def get_evaluation_results(y_true, y_pred, output_spec, return_confusion_matrix = False):
    
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
            
    results = {}
    results['# records'] = len(y_true)
            
    if output_spec.output_type.is_numeric:
        results['Spearman\'s rank correlation'] = spearmanr(y_true, y_pred)[0]
        confusion_matrix = None
    else:
    
        str_unique_labels = list(map(str, output_spec.unique_labels))
        
        if output_spec.output_type.is_binary:
            
            y_pred_classes = (y_pred >= 0.5)
            
            if len(np.unique(y_true)) == 2:
                results['AUC'] = roc_auc_score(y_true, y_pred)
            else:
                results['AUC'] = np.nan
        elif output_spec.output_type.is_categorical:
            y_pred_classes = y_pred.argmax(axis = -1)
            results['Accuracy'] = accuracy_score(y_true, y_pred_classes)
        else:
            raise ValueError('Unexpected output type: %s' % output_spec.output_type)
                    
        confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_classes, labels = np.arange(output_spec.n_unique_labels)), index = str_unique_labels,                     columns = str_unique_labels)
         
    if return_confusion_matrix:
        return results, confusion_matrix
    else:
        return results

