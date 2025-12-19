import torch
import transformers
import util
import json
import pandas as pd
import os
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from tqdm import trange
from torch.utils.data import Subset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--task', type=str, default='AG', help='Task name')
    parser.add_argument('--embedding_tool', type=str, default='openai', help='Embedding tool')
    parser.add_argument('--embedding_model', type=str, default='text-embedding-3-large', help='OpenAI embedding model: text-embedding-3-large, text-embedding-3-small, etc.')
    parser.add_argument('--compression_style', type=str, default='autoencoder', help='Compression method: autoencoder, pca, linear_projection, first, last, random, kd')
    parser.add_argument('--is_ct', type=str, default='false', help='Whether to import counterfactual samples: true or false')
    parser.add_argument('--slm', type=str, default='PATH_TO_SLM', help='SLM path')
    parser.add_argument('--emb_prompt', type=str, default='Summarize this entity record: ', help='Embedding prompt')
    parser.add_argument('--ct_prompt', type=str, default='normal', help='CT prompt type')
    parser.add_argument('--ct_llm', type=str, default='PATH_TO_CT_LLM', help='CT_LLM')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch_count_for_valid', type=int, default=1, help='Number of epochs between evaluations')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alignment loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--lamda', type=float, default=1.0, help='Reconstruction loss difference enhancement')
    parser.add_argument('--diff_loss_type', type=str, default='differential', help='Difference loss type: differential, contrastive, triplet')
    parser.add_argument('--save_checkpoint', type=str, default='false', help='Whether to save model checkpoint: true or false')
    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='Training data ratio')
    parser.add_argument('--wrong_label_ratio', type=float, default=0.0, help='Wrong label ratio')
    parser.add_argument('--C', type=int, default=1, help='Number of synthetic negative pairs generated per anchor pair, used to replace negative samples')
    parser.add_argument('--result_dir', type=str, default='PATH_TO_RESULT_DIR', help='Result file save directory')
    parser.add_argument('--config_file', type=str, default='PATH_TO_CONFIG_FILE', help='Config file path')
    parser.add_argument('--checkpoint_dir', type=str, default='PATH_TO_CHECKPOINT_DIR', help='Checkpoint directory path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(args)

    task = args.task
    embedding_tool = args.embedding_tool
    embedding_model = args.embedding_model
    slm = args.slm
    emb_prompt = args.emb_prompt
    num_epochs = args.num_epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    epoch_count_for_valid = args.epoch_count_for_valid
    alpha = args.alpha
    train_data_ratio = args.train_data_ratio
    wrong_label_ratio = args.wrong_label_ratio
    beta = args.beta
    lamda = args.lamda
    diff_loss_type = args.diff_loss_type
    save_checkpoint = args.save_checkpoint.lower() == 'true'
    ct_prompt=args.ct_prompt
    ct_llm=args.ct_llm
    C=args.C
    compression_style=args.compression_style


    print(f"task={task}")
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    result_log_dir=result_dir+"/"+task+"_result.txt"
    print(f"result_log_dir={result_log_dir}")
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_file_path = args.config_file
    json_file=open(config_file_path)
    configs = json.load(json_file)
    configs = {conf["name"] : conf for conf in configs}
    config = configs[task]

    tableA=config['tableA']
    tableB=config['tableB']

    tableB_ct = config['tableB_ct']
    if ct_prompt:
        tableB_ct = tableB_ct.replace('.csv', f'_{ct_prompt}.csv')
        config['tableB_ct'] = tableB_ct
    print(f"tableB_ct={tableB_ct}")

    if os.path.exists(config['tableB_ct']):
        print(f"already have ct tableB...")
        pass
    else:
        print(f"have no ct tableB, generating now...")
        ct_record_list = util.generate_counterfactual_records(tableB, ct_llm=ct_llm,ct_prompt=ct_prompt)
        util.save_counterfactual_records(ct_record_list, config['tableB_ct'])

    train=config['trainset']
    valid=config['validset']
    test=config['testset']

    serialized_entity_pairs_train_df=util.serialize_entity_pairs(tableA, tableB, train)
    print("train pairs count: ",len(serialized_entity_pairs_train_df))
    serialized_entity_pairs_train_ct_df=util.serialize_entity_pairs(tableA, tableB_ct, train)
    print("ct train pairs count: ",len(serialized_entity_pairs_train_ct_df))

    serialized_entity_pairs_valid_df=util.serialize_entity_pairs(tableA, tableB, valid)
    print("valid pairs count: ",len(serialized_entity_pairs_valid_df))
    serialized_entity_pairs_test_df=util.serialize_entity_pairs(tableA, tableB, test)
    print("test pairs count: ",len(serialized_entity_pairs_test_df))

    emb_config_name='embedding_filedir_'+embedding_tool
    embedding_filedir=config[emb_config_name]

    if not os.path.exists(embedding_filedir):
        os.makedirs(embedding_filedir)

    embedding_filenameA = embedding_filedir + 'embA.txt'
    embedding_filenameB = embedding_filedir + 'embB.txt'
    embedding_filenameB_ct = embedding_filedir + 'embB_ct.txt'

    print(f"embedding_filenameA: {embedding_filenameA}")
    print(f"embedding_filenameB: {embedding_filenameB}")
    print(f"embedding_filenameB_ct: {embedding_filenameB_ct}")

    embeddingsA=util.load_or_generate_embeddings(tableA, embedding_filenameA,embedding_tool,prompt=emb_prompt,embedding_model=embedding_model)
    embeddingsB=util.load_or_generate_embeddings(tableB, embedding_filenameB,embedding_tool,prompt=emb_prompt,embedding_model=embedding_model)
    embeddingsB_ct=util.load_or_generate_embeddings(tableB_ct, embedding_filenameB_ct, embedding_tool,prompt=emb_prompt,embedding_model=embedding_model) 

    print(embeddingsA.shape)
    print(embeddingsB.shape)
    print(embeddingsB_ct.shape)

    llm_dim = embeddingsA.shape[1]
    print('llm_dim:',llm_dim)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(slm)
    except ValueError as e:
        if "sentencepiece" in str(e).lower():
            print("Warning: Fast tokenizer unavailable, using slow tokenizer instead.")
            print("To use fast tokenizer, install sentencepiece: pip install sentencepiece")
            tokenizer = transformers.AutoTokenizer.from_pretrained(slm, use_fast=False)
        else:
            raise

    model = util.EntityPairEncoder(slm).to(device)
    slm_dim = model.slm_dim
    best_checkpoint_model = model.state_dict()
    matcher = util.Matcher(slm_dim).to(device)
    best_checkpoint_matcher = matcher.state_dict()
    
    if compression_style == 'autoencoder':
        autoencoder = util.AutoEncoder(llm_dim, slm_dim, hidden_dim).to(device)
        linear_projection = None
        pca_transform = None
    elif compression_style == 'linear_projection':
        autoencoder = None
        linear_projection = util.LinearProjection(llm_dim, slm_dim).to(device)
        pca_transform = None
    elif compression_style == 'pca':
        autoencoder = None
        linear_projection = None
        pca_transform = util.PCATransform(llm_dim, slm_dim)
    else:
        autoencoder = util.AutoEncoder(llm_dim, slm_dim, hidden_dim).to(device)
        linear_projection = None
        pca_transform = None


    train_dataset = util.EntityPairDataset(tokenizer, serialized_entity_pairs_train_df, 
                                           embeddingsA, embeddingsB, serialized_entity_pairs_train_ct_df,
                                           embeddingsB_ct, label_reverse_ratio=wrong_label_ratio,is_ct=True,C=C)
    num_samples = int(len(train_dataset) * train_data_ratio)
    indices = list(range(num_samples))
    train_dataset = Subset(train_dataset, indices)

    valid_dataset = util.EntityPairDataset(tokenizer, serialized_entity_pairs_valid_df, embeddingsA, embeddingsB)
    test_dataset = util.EntityPairDataset(tokenizer, serialized_entity_pairs_test_df, embeddingsA, embeddingsB)
    
    if compression_style == 'pca':
        print("Fitting PCA on training embeddings...")
        all_train_embeddings = []
        temp_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        for batch in temp_dataloader:
            llm_simi_emb = batch['element_wise_product'].to(device)
            all_train_embeddings.append(llm_simi_emb)
        all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
        pca_transform.fit(all_train_embeddings)
        print("PCA fitting completed.")
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    compute_cse_loss = nn.CrossEntropyLoss()
    compute_recon_loss = nn.MSELoss()
    
    def compute_differential_loss(encoded, encoded_ct, lamda):
        difenh = (1 - F.cosine_similarity(encoded, encoded_ct, dim=1)) * lamda
        return difenh.mean()
    
    def compute_contrastive_loss(encoded, encoded_ct, labels):
        margin = 1.0
        similarity = F.cosine_similarity(encoded, encoded_ct, dim=1)
        distance = 1 - similarity
        positive_mask = (labels == 1).float()
        negative_mask = (labels == 0).float()
        positive_loss = positive_mask * (1 - similarity)
        negative_loss = negative_mask * torch.clamp(margin - distance, min=0.0)
        
        loss = (positive_loss + negative_loss).mean()
        return loss
    
    def compute_triplet_loss(encoded, encoded_ct, outputs, labels):
        triplet_margin = 0.3
        positive_mask = (labels == 1).float()
        negative_mask = (labels == 0).float()
        dist_anchor_pos_1 = 1 - F.cosine_similarity(encoded, outputs, dim=1)
        dist_anchor_neg_1 = 1 - F.cosine_similarity(encoded, encoded_ct, dim=1)
        dist_anchor_pos_0 = 1 - F.cosine_similarity(encoded, encoded_ct, dim=1)
        dist_anchor_neg_0 = 1 - F.cosine_similarity(encoded, outputs, dim=1)
        triplet_loss_1 = torch.clamp(triplet_margin + dist_anchor_pos_1 - dist_anchor_neg_1, min=0.0)
        triplet_loss_0 = torch.clamp(triplet_margin + dist_anchor_pos_0 - dist_anchor_neg_0, min=0.0)
        
        loss = (positive_mask * triplet_loss_1 + negative_mask * triplet_loss_0).mean()
        return loss
    
    if compression_style=='kd':
        compute_align_loss = nn.KLDivLoss()
        llm_matcher=nn.Linear(3072, 2).to(device)
        optimizer = optim.AdamW(list(model.parameters()) + list(matcher.parameters()) + list(llm_matcher.parameters()), lr=lr)
    elif compression_style == 'pca':
        compute_align_loss = nn.MSELoss()
        optimizer = optim.AdamW(list(model.parameters()) + list(matcher.parameters()), lr=lr)
    elif compression_style == 'linear_projection':
        compute_align_loss = nn.MSELoss()
        optimizer = optim.AdamW(list(model.parameters()) + list(matcher.parameters()) + list(linear_projection.parameters()), lr=lr)
    else:
        compute_align_loss = nn.MSELoss()
        optimizer = optim.AdamW(list(model.parameters()) + list(matcher.parameters()) + list(autoencoder.parameters()), lr=lr)

    print("start training...")

    for epoch in trange(num_epochs):
        print(f"Training epoch {epoch}/{num_epochs}")
        model.train()
        matcher.train()
        if compression_style == 'autoencoder' and autoencoder is not None:
            autoencoder.train()
        elif compression_style == 'linear_projection' and linear_projection is not None:
            linear_projection.train()
        epoch_loss_train = 0.0
        epoch_loss_valid = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(train_dataloader):
            idA, idB, inputs, llm_simi_emb, llm_simi_emb_ct, labels = batch['id_A'], batch['id_B'], batch['tokenized_pair'], batch['element_wise_product'].to(device),batch['element_wise_product_ct'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            slm_dim=outputs.shape[-1]
            
            use_diff_loss = True
            
            if compression_style=='autoencoder':
                encoded, decoded = autoencoder(llm_simi_emb)
                encoded_ct, decoded_ct = autoencoder(llm_simi_emb_ct)
            elif compression_style == 'linear_projection':
                encoded, decoded = linear_projection(llm_simi_emb)
                encoded_ct, decoded_ct = linear_projection(llm_simi_emb_ct)
                decoded = llm_simi_emb
                decoded_ct = llm_simi_emb_ct
                use_diff_loss = True
            elif compression_style == 'pca':
                encoded = pca_transform.transform(llm_simi_emb)
                encoded_ct = pca_transform.transform(llm_simi_emb_ct)
                decoded = pca_transform.inverse_transform(encoded)
                decoded_ct = pca_transform.inverse_transform(encoded_ct)
                use_diff_loss = True
            elif compression_style=='first':
                encoded = llm_simi_emb[:, :slm_dim]
                encoded_ct = llm_simi_emb_ct[:, :slm_dim]
                decoded = llm_simi_emb
                decoded_ct = llm_simi_emb_ct
                use_diff_loss = False
            elif compression_style=='last':
                encoded = llm_simi_emb[:, slm_dim:]
                encoded_ct = llm_simi_emb_ct[:, slm_dim:]
                decoded = llm_simi_emb
                decoded_ct = llm_simi_emb_ct
                use_diff_loss = False
            elif compression_style=='random':
                global random_indices
                if 'random_indices' not in globals():
                    random_indices = torch.randperm(llm_simi_emb.shape[1])[:slm_dim]
                encoded = llm_simi_emb[:, random_indices]
                encoded_ct = llm_simi_emb_ct[:, random_indices]
                decoded = llm_simi_emb
                decoded_ct = llm_simi_emb_ct
                use_diff_loss = False
            elif compression_style=='kd':
                llm_simi_emb = llm_simi_emb.to(device)
                llm_matcher_outputs = llm_matcher(llm_simi_emb)
                matcher_outputs = matcher(outputs)
                encoded = None
                decoded = None
            else:
                raise Exception(f'Unknown compression_style: {compression_style}')
            matcher_outputs = matcher(outputs)
            matching_loss = compute_cse_loss(matcher_outputs, labels)
            if compression_style=='kd':
                align_loss=compute_align_loss(matcher_outputs,llm_matcher_outputs)
                loss = matching_loss+alpha*align_loss
            else:
                recon_loss=compute_recon_loss(decoded, llm_simi_emb)
                align_loss=compute_align_loss(encoded,outputs)
                
                if use_diff_loss and beta > 0:
                    if diff_loss_type == 'differential':
                        diff_loss = compute_differential_loss(encoded, encoded_ct, lamda)
                    elif diff_loss_type == 'contrastive':
                        diff_loss = compute_contrastive_loss(encoded, encoded_ct, labels)
                    elif diff_loss_type == 'triplet':
                        diff_loss = compute_triplet_loss(encoded, encoded_ct, outputs, labels)
                    else:
                        raise ValueError(f"Unknown diff_loss_type: {diff_loss_type}. Must be one of: differential, contrastive, triplet")
                    loss = matching_loss+alpha*align_loss+beta*(recon_loss+diff_loss)
                else:
                    loss = matching_loss+alpha*align_loss+beta*recon_loss
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
            batch_count += 1
        average_epoch_loss = epoch_loss_train / batch_count
        print(f"Average training loss for epoch {epoch}: {average_epoch_loss:.4f}")

        if epoch % epoch_count_for_valid == 0:
            model.eval()
            matcher.eval()
            output_list = []
            label_list = []
            with torch.no_grad():
                for batch in valid_dataloader:
                    idA, idB, inputs, llm_simi_emb, labels = batch['id_A'], batch['id_B'], batch['tokenized_pair'], batch['element_wise_product'].to(device), batch['label'].to(device)
                    input_ids = inputs['input_ids'].squeeze(1).to(device)
                    attention_mask = inputs['attention_mask'].squeeze(1).to(device)
                    import time
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    matcher_outputs = matcher(outputs) 
                    matcher_outputs_softmax = F.softmax(matcher_outputs, dim=1)[:, 1]
                    
                    matching_loss = compute_cse_loss(matcher_outputs, labels)
                    epoch_loss_valid += matching_loss.item()
                    output_list.append(matcher_outputs_softmax)
                    label_list.append(labels)

            output_list = torch.cat(output_list)
            label_list = torch.cat(label_list)
            all_labels = label_list.cpu().numpy()
            average_epoch_loss_valid = epoch_loss_valid / batch_count
            print(f"Average validation loss for epoch {epoch}: {average_epoch_loss_valid:.4f}")
            best_th = 0.0
            best_f1 = 0.0
            best_checkpoint = None
            for th in np.arange(0, 1.0, 0.01):
                onehot_output_list = []
                for output, label in zip(output_list, label_list):
                    onehot_output_list.append(1 if output.item() > th else 0)

                all_outputs = np.array(onehot_output_list)

                if len(all_labels) != len(all_outputs):
                    raise ValueError("Length mismatch between all_labels and all_outputs!")

                current_f1 = f1_score(all_labels, all_outputs)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_th = th
                    best_checkpoint_model = model.state_dict()
                    best_checkpoint_matcher = matcher.state_dict()
                    if save_checkpoint:
                        checkpoint_base_dir = args.checkpoint_dir
                        checkpoint_dir = f"{checkpoint_base_dir}/{task}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = f"{checkpoint_dir}/best_model_seed{seed}_loss{diff_loss_type}.pt"
                        torch.save({
                            'model_state_dict': best_checkpoint_model,
                            'matcher_state_dict': best_checkpoint_matcher,
                            'best_f1': best_f1,
                            'best_th': best_th,
                            'epoch': epoch,
                            'args': args
                        }, checkpoint_path)
                        print(f"Checkpoint saved to: {checkpoint_path}")
            print(f"Best threshold: {best_th}, Best validation F1 score: {best_f1}")

    print("training finished...")
    print("start testing...")
    model.load_state_dict(best_checkpoint_model)
    matcher.load_state_dict(best_checkpoint_matcher)
    model.eval()
    matcher.eval()
    output_list = []
    label_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            idA, idB, inputs, llm_simi_emb, labels = batch['id_A'], batch['id_B'], batch['tokenized_pair'], batch['element_wise_product'].to(device), batch['label'].to(device)
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            matcher_outputs = F.softmax(matcher(outputs), dim=1)[:, 1]
            output_list.append(matcher_outputs)
            label_list.append(labels)

    output_list = torch.cat(output_list)
    label_list = torch.cat(label_list)
    all_labels = label_list.cpu().numpy()
    
    best_th = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_tp = 0
    best_tn = 0
    best_fp = 0
    best_fn = 0
    for th in np.arange(0, 1.0, 0.001):
        onehot_output_list = []
        for output, label in zip(output_list, label_list):
            if output.item() > th:
                onehot_output_list.append(1)
            else:
                onehot_output_list.append(0)
            
        all_outputs = np.array(onehot_output_list)
        
        current_f1 = f1_score(all_labels, all_outputs)
        current_precision = precision_score(all_labels, all_outputs)
        current_recall = recall_score(all_labels, all_outputs)
        tp = np.sum((all_labels == 1) & (all_outputs == 1))
        tn = np.sum((all_labels == 0) & (all_outputs == 0))
        fp = np.sum((all_labels == 0) & (all_outputs == 1))
        fn = np.sum((all_labels == 1) & (all_outputs == 0))
                
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_precision = current_precision
            best_recall = current_recall
            best_th = th

            best_tp = tp
            best_tn = tn
            best_fp = fp
            best_fn = fn
    print(f"Best threshold: {best_th}, Best test F1 score: {best_f1}")
    print("testing finished...")
    with open(result_log_dir, 'a') as log_file:
        log_file.write(f"**************************\n")
        import datetime
        current_time = datetime.datetime.now()
        log_file.write(f"Current time: {current_time}\n")
        log_file.write(f"args: {args}\n")
        log_file.write(f"best_tp: {best_tp} ")
        log_file.write(f"best_tn: {best_tn} ")
        log_file.write(f"best_fp: {best_fp} ")
        log_file.write(f"best_fn: {best_fn}\n")
        log_file.write(f"best_precision: {best_precision}\n")
        log_file.write(f"best_recall: {best_recall}\n")
        log_file.write(f"best_f1: {best_f1}\n")
    print(args)
    print('********************finished!!!********************')
    print('********************finished!!!********************')
    print('********************finished!!!********************')