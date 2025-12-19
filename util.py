import random
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel,AutoTokenizer
import numpy as np
import csv
from scipy.spatial import distance
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import os



lm_map={
    'bge_m3':'PATH_TO_BGE_M3',
    'bge_large':'PATH_TO_BGE_LARGE',
    'e5_mistral':'PATH_TO_E5_MISTRAL',
    'roberta_large':'PATH_TO_ROBERTA_LARGE',
    'deberta_large':'PATH_TO_DEBERTA_LARGE',
    'gte_qwen':'PATH_TO_GTE_QWEN',
    'NV-Embed-v2':'PATH_TO_NV_EMBED_V2'
}


class ABCVectorPairing:
    def __init__(self):
        pass

    def index(self, embedding_matrix):
        pass

    def query(self, embedding_matrix):
        pass

class ExactTopKVectorPairing(ABCVectorPairing):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def index(self, embedding_matrix_for_indexing):
        self.embedding_matrix_for_indexing = embedding_matrix_for_indexing

    def query(self, embedding_matrix_for_querying, K=None):
        if K is None:
            K = self.K

        all_pair_cosine_similarity_matrix = 1 - distance.cdist(embedding_matrix_for_querying, self.embedding_matrix_for_indexing, metric="cosine")
        topK_indices_each_row = np.argsort(-all_pair_cosine_similarity_matrix)[:, :K]

        return topK_indices_each_row

def compute_similarity(embedding_matrix_for_querying, embedding_matrix_for_indexing):
    all_pair_cosine_similarity_matrix = 1 - distance.cdist(embedding_matrix_for_querying,
                                                           embedding_matrix_for_indexing, metric="cosine")
    return all_pair_cosine_similarity_matrix

def read_csv_to_string_list(file_path):
    string_list = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        rows = list(csv_reader)
        first_row = rows[0]
        keys = first_row.keys()
        if 'content' in keys:
            for row in rows:
                string_list.append(row['content'])
        else:
            for row in rows:
                row_string = ' '.join([f"COL {column} VAL {value}" for column, value in row.items() if column != 'id'])
                string_list.append(row_string)

    return string_list


def get_embeddings_from_openai(text_list, channel='', prompt='', model='text-embedding-3-large'):
    print("get_embeddings_from_openai...")
    print("channel is",channel)
    print(f"Total texts to process: {len(text_list)}")
    print(f"Using embedding model: {model}")
   
    import openai
    import time
    from tqdm import tqdm
    
    if prompt is None or prompt == '':
        prompt = 'none'
    
    # Get API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running.")
    openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    embeddings = []
    start_time = time.time()
    
    for i, text in enumerate(tqdm(text_list, desc="Generating embeddings")):
        if text is None:
            text = ''
        if prompt is None:
            prompt = 'none'
        response = openai.embeddings.create(
            model=model,
            input=prompt+text
        )
        embedding=response.data[0].embedding
        embeddings.append(embedding)
        
        if (i + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / (i + 1)
            remaining_items = len(text_list) - (i + 1)
            estimated_remaining_time = avg_time_per_item * remaining_items
            print(f"\nProgress: {i+1}/{len(text_list)} | "
                  f"Elapsed: {elapsed_time/60:.1f}min | "
                  f"Estimated remaining: {estimated_remaining_time/60:.1f}min")

    total_time = time.time() - start_time
    print(f"\nCompleted! Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"Average time per item: {total_time/len(text_list):.2f} seconds")

    return embeddings

def get_embeddings_from_openai1(text_list, channel='', prompt=''):
    print("get_embeddings_from_openai...")
    print("channel is",channel)
    
    if prompt is None or prompt == '':
        prompt = 'none'
    
    if channel == 'openai':
        url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running.")
    else:
        raise Exception("channel not defined!")


    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {key}'
    }

    embeddings = []

    for text in text_list:
        if text is None:
            text = ''
        if prompt is None:
            prompt = 'none'
        data = {
            "model": "text-embedding-3-small",
            "input": prompt+text
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            embedding = response.json()['data'][0]["embedding"]
            embeddings.append(embedding)
        else:
            print(f"Failed to get embedding for text: {text}")
            print('response.status_code ==', response.status_code)
            while response.status_code != 200:
                response = requests.post(url, headers=headers, json=data)
            embedding = response.json()['data'][0]["embedding"]
            embeddings.append(embedding)

    return embeddings

def get_embeddings_from_sentence_transformers(text_list,lm,prompt):
    print("get_embeddings_from_sentence_transformers...")
    
    if prompt is None or prompt == '':
        prompt = 'none'
    
    model = SentenceTransformer(lm_map[lm])
    embeddings = []
    for text in text_list:
        embedding = model.encode(text, normalize_embeddings=True,prompt=prompt)
        embeddings.append(embedding)
    return embeddings

def get_embeddings_from_huggingface(text_list,lm):

    try:
        tokenizer = AutoTokenizer.from_pretrained(lm_map[lm])
    except ValueError as e:
        if "sentencepiece" in str(e).lower():
            print("Warning: Fast tokenizer unavailable, using slow tokenizer instead.")
            print("To use fast tokenizer, install sentencepiece: pip install sentencepiece")
            tokenizer = AutoTokenizer.from_pretrained(lm_map[lm], use_fast=False)
        else:
            raise
    model = AutoModel.from_pretrained(lm_map[lm])
    model.eval()

    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output[0][:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def get_embeddings_from_nvembed(text_list,lm):
    pass


def get_ct_record(ctprompt, llm,anchor_record):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": ctprompt+anchor_record,
            }
        ],
        model=llm,
    )

    ct_record = response.choices[0].message.content

    return ct_record


def get_ct_records_from_openai(text_list,llm,ct_prompt):
    ct_record_list=[]
    for record in text_list:
        attempts = 0
        while attempts < 5:
            try:
                ct_record = get_ct_record(ct_prompt,llm, record)
                ct_record = ct_record.replace("\n", "").replace("\t", "")
                break
            except Exception as e:
                attempts += 1
                print(f"Failed to get ct_record, error: {e}. Attempt: {attempts}")
        ct_record_list.append(ct_record)
    return ct_record_list
        



def save_file(embeddings, filename):
    with open(filename, 'w') as f:
        for vector in embeddings:
            np.savetxt(f, [vector], fmt='%.8f')

def read_file_as_tensor(filename):
    loaded_vectors = []
    with open(filename, 'r') as file:
        for line in file:
            loaded_vectors.append(np.fromstring(line.strip(), sep=' '))

    tensor = torch.tensor(loaded_vectors, dtype=torch.float32)
    return tensor

def generate_candidate(embA, embB, pairing_model):
    pairing_model.index(embB)
    topK_neighbors = pairing_model.query(embA)
    print(topK_neighbors)
    print(len(topK_neighbors))
    print(len(topK_neighbors[0]))

    candi_pairs=[]
    for query_id, row in enumerate(np.array(topK_neighbors)):
        for record_id in row:
            candi_pairs.append((query_id, record_id))

    candi_pair_df = pd.DataFrame(candi_pairs, columns=['ltable_id', 'rtable_id'])
    print(candi_pair_df)

    return candi_pair_df

def get_groundtruth_matches(train_df, valid_df, test_df):

    train_positive = train_df[train_df['label'] == 1]
    valid_positive = valid_df[valid_df['label'] == 1]
    test_positive = test_df[test_df['label'] == 1]

    positive_rows = pd.concat([train_positive, valid_positive, test_positive])

    matches_df = positive_rows[['ltable_id', 'rtable_id']]

    return matches_df

def evaluate_block(lenA, lenB, candidate_pairs, groundtruth_matches):
    matches_num=len(groundtruth_matches)
    candidate_num = len(candidate_pairs)
    merged_df = pd.merge(left=groundtruth_matches, right=candidate_pairs, how='inner', on=['ltable_id', 'rtable_id'])
    merge_num = len(merged_df)
    recall = merge_num / matches_num
    recall = int(recall * 10000) / 10000
    cssr = candidate_num / (lenA * lenB)
    rr = 1 - cssr
    rr = int(rr * 10000) / 10000

    statistics_dict = {
        "candidate_size": candidate_num,
        "left_num": lenA,
        "right_num": lenB,
        "rr": rr,
        "recall": recall
    }

    return statistics_dict



def evaluate_matching(similarity_matrix, eval_data_df, th=None):
    y_true = []
    y_pred = []
    if th==None:
        best_th=0
        best_f1=0
        for th in np.arange(0, 1, 0.001):
            for index, row in eval_data_df.iterrows():
                l_id=row['ltable_id']
                r_id=row['rtable_id']
                label=row['label']
                similarity = similarity_matrix[l_id][r_id]
                y_pred.append(1 if similarity>th else 0)
                y_true.append(label)
            f1 = f1_score(y_true, y_pred)
            if f1>best_f1:
                best_th=th
                best_f1=f1
        return best_f1, best_th

    else:
        for index, row in eval_data_df.iterrows():
            l_id = row['ltable_id']
            r_id = row['rtable_id']
            label = row['label']
            similarity = similarity_matrix[l_id][r_id]
            y_pred.append(1 if similarity > th else 0)
            y_true.append(label)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [precision,recall,f1], th


def serialize_entity(table_path):
    serialized_entities = []
    
    with open(table_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        
        for row in reader:
            serialized = ""
            for header, value in zip(headers[1:], row[1:]):
                serialized += f"COL {header} VAL {value} "
            serialized = serialized.strip()
            serialized_entities.append(serialized)
    
    return serialized_entities

def serialize_entity_pairs(tableA_path, tableB_path, label_dir=None):
    label_df = pd.read_csv(label_dir)
    serialized_A = serialize_entity(tableA_path)
    print(f"serialized_A length: {len(serialized_A)}")
    if "_ct" not in tableB_path:
        serialized_B = serialize_entity(tableB_path)
    else:
        serialized_B = []
        with open(tableB_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                serialized_B.append(row['content'])
    print(f"serialized_B length: {len(serialized_B)}")

    dfA = pd.read_csv(tableA_path)
    dfB = pd.read_csv(tableB_path)

    
    result_list = []
    
    for _, row in label_df.iterrows():
        id_A = row['ltable_id']
        id_B = row['rtable_id']
        label = row['label']
        
        ser_A = serialized_A[dfA.index[dfA['id'] == id_A][0]] if any(dfA['id'] == id_A) else ""
        ser_B = serialized_B[dfB.index[dfB['id'] == id_B][0]] if any(dfB['id'] == id_B) else ""

        
        result_list.append({
            'id_A': id_A,
            'id_B': id_B,
            'ser_A': ser_A,
            'ser_B': ser_B,
            'label': label
        })
    
    result = pd.DataFrame(result_list)


    
    return result


def generate_counterfactual_records(tableB, ct_llm=None, ct_prompt=None):
    ct_prompt_dict = {
        "normal": "I'll give you a record whose attributes and attribute values are separated by COL and VAL. Please generate a new record by changing one or some of the attribute values. The generated record needs to be significantly different from the original record, but maintain the same format. Please output the generated record directly without adding any other content. The record is as follows: ",
        "oneunimportattr": "I'll give you a record whose attributes and attribute values are separated by COL and VAL. Please generate a new record by changing only one unimportant (non-decisive) attribute such as version, price, or description, while keeping important (decisive) attributes such as title or name unchanged. The generated record should be slightly different from the original record, but maintain the same format. Please output the generated record directly without adding any other content. The record is as follows: "
    }
    
    tableB_ct = tableB.replace('.csv', '_ct.csv')
    print(f"tableB_ct={tableB_ct}")
    print(f"ct_prompt={ct_prompt_dict[ct_prompt]}")
    print(f"llm={ct_llm}")

    serialized_B = serialize_entity(tableB)
    if ct_llm in ['Qwen/Qwen2.5-7B-Instruct','gpt-4o-mini', 'deepseek-chat', 'THUDM/glm-4-9b-chat']:
        ct_record_list = get_ct_records_from_openai(text_list=serialized_B, llm=ct_llm, ct_prompt=ct_prompt_dict[ct_prompt])
    else:
        raise Exception("ct_llm definition error")
    
    return ct_record_list

def save_counterfactual_records(ct_record_list, tableB_ct_dir):
    df_ct_records = pd.DataFrame({'id': range(0, len(ct_record_list)), 'content': ct_record_list})

    df_ct_records.to_csv(tableB_ct_dir, index=False)



def load_or_generate_embeddings(table, embedding_filename, embedding_tool, prompt, embedding_model='text-embedding-3-large'):
    if os.path.exists(embedding_filename):
        print('creating embeddings from existing file...')
        embeddings=read_file_as_tensor(embedding_filename)
    else:
        print('no embeddings found, encoding entities...')
        text_list = read_csv_to_string_list(table)
        print(f"len(text_list)={len(text_list)}")
        if len(text_list)==0:
            raise Exception(f"len(text_list)=0!!!")
        if embedding_tool=="openai":
            embeddings=get_embeddings_from_openai(text_list, channel = 'openai', prompt=prompt, model=embedding_model)
        elif embedding_tool in ["bge_m3","bge_large","deberta_large","roberta_large",
                                "e5_mistral","gte_qwen","nvidia/NV-Embed-v2"]:
            embeddings=get_embeddings_from_sentence_transformers(text_list,lm=embedding_tool,prompt=prompt)
        else:
            raise Exception("Unknown embedding tool")
        print('saving embeddings...')
        save_file(embeddings, embedding_filename)
        embeddings=read_file_as_tensor(embedding_filename)

    return embeddings


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class EntityPairDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, df ,embeddingsA, embeddingsB, 
                 df_ct=None, embeddingsB_ct=None, max_length=512, label_reverse_ratio=0,is_ct=False, C=1):
        self.df = df
        self.embeddingsA = embeddingsA
        self.embeddingsB = embeddingsB
        self.is_ct=is_ct
        if embeddingsB_ct is not None:
            self.is_ct=True
            self.embeddingsB_ct = embeddingsB_ct
            self.df_ct=df_ct
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_reverse_ratio = label_reverse_ratio
        self.C = C
        self.preprocessed_data = self.preprocess_data()

    def preprocess_data(self):
        preprocessed_data = []
        if self.is_ct:
            total_length = len(self.df)
            reverse_length = int(total_length * self.label_reverse_ratio)
            
            anchor_to_negatives = {}
            
            anchor_id_as = set()
            for idx, (index, row) in enumerate(self.df.iterrows()):
                label = torch.tensor(row['label'], dtype=torch.long)
                if label.item() == 1:
                    id_a = row['id_A']
                    anchor_id_as.add(id_a)
                    if id_a not in anchor_to_negatives:
                        anchor_to_negatives[id_a] = []
            
            for idx, (index, row) in enumerate(self.df.iterrows()):
                label = torch.tensor(row['label'], dtype=torch.long)
                if label.item() == 0:
                    id_a = row['id_A']
                    if id_a in anchor_to_negatives:
                        anchor_to_negatives[id_a].append(idx)
            
            negative_indices_to_replace = set()
            total_replace_count = 0
            
            for id_a, negative_indices in anchor_to_negatives.items():
                num_to_replace = min(self.C, len(negative_indices))
                for i in range(num_to_replace):
                    negative_indices_to_replace.add(negative_indices[i])
                    total_replace_count += 1
            
            num_anchor_pairs = len(anchor_id_as)
            if num_anchor_pairs > 0:
                avg_replace = total_replace_count / num_anchor_pairs
                print(f"Negative sample replacement stats: total anchors={num_anchor_pairs}, C={self.C}, "
                      f"total replacements={total_replace_count}, avg replacements per anchor={avg_replace:.2f}")
            
            for idx, ((index, row), (_, row_ct)) in enumerate(zip(self.df.iterrows(), self.df_ct.iterrows())):
                id_a = row['id_A']
                if id_a >= len(self.embeddingsA):
                    print(f"Warning: id_A {id_a} is out of bounds for embeddingsA with size {len(self.embeddingsA)}")
                    raise Exception("...")
                embA = self.embeddingsA[id_a]
                embB = self.embeddingsB[row['id_B']]
                embB_ct = self.embeddingsB[row_ct['id_B']]

                combined_text = row['ser_A'] + ' ' + row['ser_B']
                combined_text_ct = row_ct['ser_A'] + ' ' + row_ct['ser_B']

                tokenized_pair = self.tokenizer(combined_text, 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=int(self.max_length),
                                        return_tensors="pt")
                tokenized_pair_ct = self.tokenizer(combined_text_ct, 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=int(self.max_length),
                                        return_tensors="pt")
                
                element_wise_product = embA * embB
                element_wise_product_ct = embA * embB_ct

                label = torch.tensor(row['label'], dtype=torch.long)
                
                should_replace = False
                if label.item() == 0 and idx in negative_indices_to_replace:
                    should_replace = True

                if should_replace:
                    tokenized_pair = tokenized_pair_ct
                    element_wise_product = element_wise_product_ct

                if reverse_length > 0:
                    label = 1 - label
                    reverse_length -= 1
                preprocessed_data.append((row['id_A'], row['id_B'], 
                                        tokenized_pair, element_wise_product, element_wise_product_ct,  label))
                
            return preprocessed_data 
        else:
            for (index, row) in self.df.iterrows():
                embA = self.embeddingsA[row['id_A']]
                embB = self.embeddingsB[row['id_B']]

                combined_text = row['ser_A'] + ' ' + row['ser_B']

                tokenized_pair = self.tokenizer(combined_text, 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=int(self.max_length),
                                        return_tensors="pt")
                
                element_wise_product = embA * embB

                label = torch.tensor(row['label'], dtype=torch.long)

                preprocessed_data.append((row['id_A'], row['id_B'], 
                                        tokenized_pair, element_wise_product, label))
                
            return preprocessed_data 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.is_ct:
            return {
                'id_A': self.preprocessed_data[idx][0],
                'id_B': self.preprocessed_data[idx][1],
                'tokenized_pair': self.preprocessed_data[idx][2],
                'element_wise_product': self.preprocessed_data[idx][3],
                'element_wise_product_ct': self.preprocessed_data[idx][4],
                'label': self.preprocessed_data[idx][5]
            }
        else:
            return {
                'id_A': self.preprocessed_data[idx][0],
                'id_B': self.preprocessed_data[idx][1],
                'tokenized_pair': self.preprocessed_data[idx][2],
                'element_wise_product': self.preprocessed_data[idx][3],
                'label': self.preprocessed_data[idx][4]
            }

class EntityPairEncoder(nn.Module):
    def __init__(self, model_name='microsoft/deberta-base'):
        super(EntityPairEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.slm_dim = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        last_hidden_state = outputs.last_hidden_state
        
        attention_mask = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled
    
class Matcher(nn.Module):
    def __init__(self, input_dim):
        super(Matcher, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
    

class AutoEncoder(nn.Module):
    def __init__(self, llm_dim, slm_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, slm_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(slm_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded


class LinearProjection(nn.Module):
    def __init__(self, llm_dim, slm_dim):
        super(LinearProjection, self).__init__()
        self.projection = nn.Linear(llm_dim, slm_dim)
    
    def forward(self, x):
        encoded = self.projection(x)
        decoded = x
        return encoded, decoded


class PCATransform:
    def __init__(self, llm_dim, slm_dim):
        from sklearn.decomposition import PCA
        self.slm_dim = slm_dim
        self.pca = None
        self.fitted = False
    
    def fit(self, embeddings):
        import numpy as np
        from sklearn.decomposition import PCA
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = np.array(embeddings)
        
        if len(embeddings_np.shape) == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        elif len(embeddings_np.shape) > 2:
            embeddings_np = embeddings_np.reshape(-1, embeddings_np.shape[-1])
        
        n_samples, n_features = embeddings_np.shape
        max_components = min(n_samples, n_features, self.slm_dim)
        
        if max_components < self.slm_dim:
            print(f"Warning: Training samples ({n_samples}) < target dimension ({self.slm_dim}). "
                  f"Using {max_components} components instead.")
        
        self.pca = PCA(n_components=max_components)
        self.pca.fit(embeddings_np)
        self.fitted = True
    
    def transform(self, x):
        import numpy as np
        import torch
        
        if isinstance(x, torch.Tensor):
            device = x.device
            x_np = x.detach().cpu().numpy()
            is_tensor = True
        else:
            x_np = np.array(x)
            is_tensor = False
        
        original_shape = x_np.shape
        if len(original_shape) > 2:
            x_np = x_np.reshape(-1, original_shape[-1])
        
        transformed = self.pca.transform(x_np)
        
        actual_dim = transformed.shape[-1]
        if actual_dim < self.slm_dim:
            padding = np.zeros((transformed.shape[0], self.slm_dim - actual_dim))
            transformed = np.concatenate([transformed, padding], axis=1)
        
        if len(original_shape) > 2:
            transformed = transformed.reshape(original_shape[0], -1, transformed.shape[-1])
        
        if is_tensor:
            transformed = torch.from_numpy(transformed).float().to(device)
        
        return transformed
    
    def inverse_transform(self, x):
        import numpy as np
        import torch
        
        if isinstance(x, torch.Tensor):
            device = x.device
            x_np = x.detach().cpu().numpy()
            is_tensor = True
        else:
            x_np = np.array(x)
            is_tensor = False
        
        original_shape = x_np.shape
        if len(original_shape) > 2:
            x_np = x_np.reshape(-1, original_shape[-1])
        
        actual_pca_dim = self.pca.n_components_
        if x_np.shape[-1] > actual_pca_dim:
            x_np = x_np[:, :actual_pca_dim]
        
        reconstructed = self.pca.inverse_transform(x_np)
        
        if len(original_shape) > 2:
            reconstructed = reconstructed.reshape(original_shape[0], -1, reconstructed.shape[-1])
        
        if is_tensor:
            reconstructed = torch.from_numpy(reconstructed).float().to(device)
        
        return reconstructed




if __name__ == '__main__':
    pass