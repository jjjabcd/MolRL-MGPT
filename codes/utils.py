import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc import Evaluator
from tqdm import tqdm

import threading

from vocabulary import SMILESTokenizer, read_vocabulary

def randomize_smiles(smi):
    """
    SMILES 문자열을 랜덤하게 변형하되, 에러 처리를 추가
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:  # SMILES 파싱 실패
            return smi
        
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if new_mol is None:  # 변환 실패
            return smi
            
        return Chem.MolToSmiles(new_mol, canonical=False, doRandom=True)
    except:
        # 어떤 에러가 발생하더라도 원본 SMILES 반환
        return smi

def likelihood(model, seqs):
    nll_loss = nn.NLLLoss(reduction="none")
    seqs = seqs.cuda()
    logits, _ = model(seqs[:, :-1])
    log_probs = logits.log_softmax(dim=2)
    return nll_loss(log_probs.transpose(1, 2), seqs[:, 1:]).sum(dim=1)


@torch.no_grad()
def sample_SMILES(model, voc, n_mols=100, block_size=100, temperature=1.0, top_k=10):
    nll_loss = nn.NLLLoss(reduction="none")
    codes = torch.zeros((n_mols, 1), dtype=torch.long).to("cuda")
    codes[:] = voc["^"]
    nlls = torch.zeros(n_mols).to("cuda")

    model.eval()
    for k in range(block_size - 1):
        logits, _ = model(codes)  
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, k=top_k)
        # apply softmax to convert to probabilities
        probs = logits.softmax(dim=-1)
        log_probs = logits.log_softmax(dim=1)
        # sample from the distribution
        code_i = torch.multinomial(probs, num_samples=1)
        # print(probs)
        # append to the sequence and continue
        codes = torch.cat((codes, code_i), dim=1)

        nlls += nll_loss(log_probs, code_i.view(-1))
        if code_i.sum() == 0:
            break

    # codes = codes
    smiles = []
    Tokenizer = SMILESTokenizer()
    for i in range(n_mols):
        tokens_i = voc.decode(np.array(codes[i, :].cpu()))
        smiles_i = Tokenizer.untokenize(tokens_i)
        smiles.append(smiles_i)

    return smiles, codes, nlls
    

def model_validity(model, vocab_path, n_mols=100, block_size=100):
    evaluator = Evaluator(name = 'Validity')
    voc = read_vocabulary(vocab_path)
    smiles, _, _ = sample_SMILES(model, voc=voc, n_mols=n_mols, block_size=block_size, top_k=10)
    return evaluator(smiles)


def calc_fingerprints(smiles_list):
    """
    SMILES 리스트에서 분자 지문을 계산하고 유효하지 않은 SMILES는 처리
    """
    mols = []
    valid_smiles = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:  # 유효한 분자만 추가
            mols.append(mol)
            valid_smiles.append(smi)
    
    if not mols:  # 모든 분자가 유효하지 않은 경우
        return None, None
        
    try:
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in mols]
        return fps, valid_smiles
    except Exception as e:
        print(f"Error in calculating fingerprints: {str(e)}")
        return None, None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)