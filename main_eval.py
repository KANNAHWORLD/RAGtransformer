# Huggingface
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk

# PyTorch
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F

#Misc
from itertools import chain
import tqdm
import os
import bisect
import argparse
import shlex

# Disk locations
VALIDATION_DISK_NAME =  "./tokenized/validation/"
TRAIN_DISK_NAME = "./tokenized/train/"

# Huggingface hub data
DATASET_NAME = "allenai/quac"
BASE_MODEL = 'distilbert/distilbert-base-uncased' #"facebook/bart-large" "bert-base-uncased"
CHECKPOINT = 'checkpoint/distilbert/distilbert-base-uncased_0_4_0.0001'

# Control Flow
LOAD_PROCESS_DATASET = True  # Load and process new data from the huggingface hub
TRAIN_MODEL = True           # Train a new model
EVALUATE = True               # Evaluate the model on evaluation set
MANUAL_EVALUATE = False       # Evaluate the model manually
LOAD_CHECKPOINT = False       # Load a checkpoint

# Hyperparams
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 3
MAX_SEQUENCE_LENGTH = 384 # Model dependent, 384 for distilbert
STRIDE = 50

# Machine Specific Parameters
DEVICE = torch.device("cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

print(f"Using device: {DEVICE}")

# Making sure proper directories are created beforehand
if not os.path.exists(VALIDATION_DISK_NAME): os.makedirs(VALIDATION_DISK_NAME)
if not os.path.exists(TRAIN_DISK_NAME): os.makedirs(TRAIN_DISK_NAME)


def preprocess_training(dataset):

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, force_download=False)
    
    final_dataset = []

    for datapoint in tqdm.tqdm(dataset):
        questions = [q.strip() for q in datapoint["questions"]]
        # print(questions)
        context = datapoint["context"].replace('CANNOTANSWER', '')
        for idx, question in enumerate(questions):
            input_ = tokenizer(question, 
                            context, 
                            max_length=MAX_SEQUENCE_LENGTH, 
                            truncation="only_second", 
                            stride=STRIDE,
                            return_overflowing_tokens=True, 
                            return_offsets_mapping=True, 
                            padding="max_length"
                            )

            offset_mapping = input_.pop("offset_mapping")
            sample_map = input_.pop("overflow_to_sample_mapping")
            answers = datapoint["orig_answers"]
            start_idx = answers['answer_starts'][idx]
            end_idx = start_idx + len(answers['texts'][idx])
            
            for sample in range(len(sample_map)):
                sequence_id = input_.sequence_ids(sample)
                i = 0

                # Finding the start and end of the context
                while sequence_id[i] != 1:
                    i += 1
                start_context = i
                while sequence_id[i] == 1:
                    i += 1
                end_context = i - 1

                start_answer_loc = bisect.bisect_left(offset_mapping[sample][start_context:end_context], start_idx, key=lambda x: x[0]) + start_context
                end_answer_loc = bisect.bisect_right(offset_mapping[sample][start_context:end_context], end_idx, key=lambda x: x[1]) + start_context

                # In the case the answer is not contained at all within this context
                if start_answer_loc >= end_context or end_answer_loc <= start_context or answers['texts'][idx].upper() == 'CANNOTANSWER':
                    start_answer_loc = 0
                    end_answer_loc = 0
                else:
                    if offset_mapping[sample][end_answer_loc][0] > end_idx:
                        end_answer_loc -= 1
                
                final_dataset.append({
                    'input_ids': input_['input_ids'][sample],
                    'attention_mask': input_['attention_mask'][sample],
                    'start_positions': start_answer_loc,
                    'end_positions': end_answer_loc
                })

                # print('Tokenized_text: ', tokenizer.decode(input_.input_ids[sample]))
                # print('start toekn, end toekn', start_answer_loc, end_answer_loc)
                # print('original text index', start_idx, end_idx)
                # print('mapping of char index to token index', offset_mapping[sample][start_answer_loc], offset_mapping[sample][end_answer_loc])
                # print('answer to question', answers['texts'][idx])
                # input()

            # print("-------------------")
            # print(len(offset_mapping), offset_mapping[1:2])
            # print("-------------------")
            # print(len(sample_map), sample_map)
            # print("-------------------")
            # print(input_.sequence_ids(1))

    return final_dataset

def compute_validation_loss(model, validation_data):
    model.eval()
    loss = 0

    validation_data = DataLoader(validation_data, batch_size=BATCH_SIZE)
    for batch in tqdm.tqdm(validation_data):
        
        batch['input_ids'] = torch.stack(batch['input_ids']).to(DEVICE).T
        batch['attention_mask'] = torch.stack(batch['attention_mask']).to(DEVICE).T
        batch['start_positions'] = torch.tensor(batch['start_positions']).to(DEVICE)
        batch['end_positions'] = torch.tensor(batch['end_positions']).to(DEVICE)

        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start = torch.argmax(start_logits, dim=1)
        end = torch.argmax(end_logits, dim=1)
        # print(start, end, batch['start_positions'], batch['end_positions'])
        se = start-batch['start_positions']
        le = end-batch['end_positions']
        loss += torch.sum(torch.pow((le-se), 2), dim=0)

    print(loss)
    print('average mse_loss = ', loss/len(validation_data))


    model.train()

def manual_compute_validation_loss(model, validation_data):
    model.eval()
    loss = 0
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, force_download=False)

    validation_data = DataLoader(validation_data, batch_size=1)
    for batch in tqdm.tqdm(validation_data):
        
        batch['input_ids'] = torch.stack(batch['input_ids']).to(DEVICE).T
        batch['attention_mask'] = torch.stack(batch['attention_mask']).to(DEVICE).T
        batch['start_positions'] = torch.tensor(batch['start_positions']).to(DEVICE)
        batch['end_positions'] = torch.tensor(batch['end_positions']).to(DEVICE)

        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start = torch.argmax(start_logits, dim=1)
        end = torch.argmax(end_logits, dim=1)
        # print(batch['input_ids'])
        text = tokenizer.decode(batch['input_ids'][0])
        print(text)
        pred = tokenizer.decode(batch['input_ids'][0][start:end])
        print(pred)
        act = tokenizer.decode(batch['input_ids'][0][batch['start_positions']:batch['end_positions']])
        print(act)
        
        print('pred:  ', start, end)
        print('actual:', batch['start_positions'], batch['end_positions'])
        # print(start, end, batch['start_positions'], batch['end_positions'])
        se = start-batch['start_positions']
        le = end-batch['end_positions']
        loss += torch.sum(torch.pow((le-se), 2), dim=0)
        input()

    print(loss)


    model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input')
    
    args = parser.parse_args()
    file = args.input

    def parse_file(filename):
        with open(filename, 'r') as f:
            args = shlex.split(f.read())
        return args

    parser = argparse.ArgumentParser()
    ## parse the arguments for ep, weighted, atoms, sparsity, seed, epochs
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')

    args = parser.parse_args(parse_file(file))

    BATCH_SIZE = args.batch
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs

    if LOAD_PROCESS_DATASET:
        train_data = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)
        validation_data = load_dataset(DATASET_NAME, split='validation', trust_remote_code=True)

        train_data = preprocess_training(train_data)
        validation_data = preprocess_training(validation_data)

        train_data = Dataset.from_list(train_data)
        validation_data = Dataset.from_list(validation_data)

        train_data.save_to_disk(TRAIN_DISK_NAME)
        validation_data.save_to_disk(VALIDATION_DISK_NAME)

    else:
        train_data = Dataset.load_from_disk(TRAIN_DISK_NAME)
        validation_data = Dataset.load_from_disk(VALIDATION_DISK_NAME)


    if not LOAD_CHECKPOINT:
        model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)
        model.to(DEVICE)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT)
        model.to(DEVICE)

    if TRAIN_MODEL:   
        model.train()
        
        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        learning_rate = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        for epoch in range(EPOCHS):
            for batch in tqdm.tqdm(train_data_loader):
                
                # print(batch)
                batch['input_ids'] = torch.stack(batch['input_ids']).to(DEVICE).T
                batch['attention_mask'] = torch.stack(batch['attention_mask']).to(DEVICE).T
                batch['start_positions'] = torch.tensor(batch['start_positions']).to(DEVICE)
                batch['end_positions'] = torch.tensor(batch['end_positions']).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(**batch)

                loss = outputs.loss
                loss.backward()
                optimizer.step()

            compute_validation_loss(model, validation_data)
            
            learning_rate.step()

            # Saving model checkpoint
            MODEL_PATH = os.path.join(os.getcwd(), f'checkpoint/{BASE_MODEL}_EPOCH{epoch}_BATCH{BATCH_SIZE}_LR{LEARNING_RATE}')
            if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
            model.save_pretrained(MODEL_PATH)

    # Begin model evaluation

    if MANUAL_EVALUATE:
        manual_compute_validation_loss(model, validation_data)

    # file = open(MODEL_PATH + '/output.txt', 'w')
    # print(total_loss, file=file)
    # file.close()


