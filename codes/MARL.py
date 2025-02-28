import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

# rdkit
from rdkit import Chem, DataStructs

from model import GPT, GPTConfig
from vocabulary import read_vocabulary
from utils import set_seed, sample_SMILES, likelihood, to_tensor, calc_fingerprints
from scoring_function import get_scores, int_div

class MARL_trainer():

    def __init__(self, logger, configs):
        self.writer = logger
        self.model_type = configs.model_type
        self.device = configs.device
        self.oracle = configs.oracle.strip()
        self.num_agents = configs.num_agents
        self.prior_path = configs.prior_path
        self.voc = read_vocabulary(configs.vocab_path)
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        self.sigma1 = configs.sigma1
        self.sigma2 = configs.sigma2
        # experience replay
        self.memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps"])
        self.memory_size = configs.memory_size
        self.replay = configs.replay
        # penalize similarity
        self.sim_penalize = configs.sim_penalize
        self.sim_thres = configs.sim_thres
        

    def _memory_update(self, samples, scores, seqs):
        """메모리 업데이트 함수 수정"""
        for i in range(self.num_agents):
            smiles_i = samples[i]
            if not smiles_i:  # 빈 리스트 체크
                continue
            
            # fingerprint 계산
            fp, valid_smiles = calc_fingerprints(smiles_i)
            if fp is None or not valid_smiles:  # None 체크 추가
                continue
            
            # seqs의 크기 확인
            max_seq_idx = len(seqs[i]) - 1
            
            # 유효한 SMILES에 대한 scores 필터링 (seqs 크기 제한 추가)
            valid_indices = [j for j, smi in enumerate(smiles_i) if smi in valid_smiles and j <= max_seq_idx]
            if not valid_indices:  # 유효한 인덱스가 없으면 건너뛰기
                continue
            
            # scores가 리스트가 아닌 경우 처리
            if isinstance(scores[i], (float, int)):
                filtered_scores = [scores[i]] * len(valid_indices)
            else:
                filtered_scores = [scores[i][j] for j in valid_indices]
            
            # seqs를 CPU로 이동
            if isinstance(seqs[i][0], torch.Tensor):
                filtered_seqs = [seq.cpu().numpy() for seq in [seqs[i][j] for j in valid_indices]]
            else:
                filtered_seqs = [seqs[i][j] for j in valid_indices]
            
            # DataFrame 생성
            new_data = pd.DataFrame({
                "smiles": valid_smiles[:len(valid_indices)],  # valid_indices 길이에 맞춤
                "scores": filtered_scores,
                "seqs": filtered_seqs,
                "fps": fp[:len(valid_indices)]  # valid_indices 길이에 맞춤
            })
            
            # 메모리에 추가
            if self.memory is None:
                self.memory = new_data
            else:
                self.memory = pd.concat([self.memory, new_data], ignore_index=True)
            
            # 메모리 크기 제한
            if len(self.memory) > self.memory_size:
                self.memory = self.memory.nlargest(self.memory_size, "scores")
            
        return samples, scores, seqs


    def train(self):
        # oracle 디렉토리 초기화
        if os.path.exists('./oracle'):
            shutil.rmtree('./oracle')
        os.makedirs('./oracle', exist_ok=True)

        if not os.path.exists(f'outputs/{self.oracle}'):
            os.makedirs(f'outputs/{self.oracle}')

        if self.model_type == "gpt":
            prior_config = GPTConfig(self.voc.__len__(), n_layer=8, n_head=8, n_embd=256, block_size=128)
            prior = GPT(prior_config).to(self.device)
            agents = []
            optimizers = []
            for i in range(self.num_agents):
                agents.append(GPT(prior_config).to(self.device))
                optimizers.append(agents[i].configure_optimizers(weight_decay=0.1, 
                                                                learning_rate=self.learning_rate, 
                                                                betas=(0.9, 0.95)))
        
        prior.load_state_dict(torch.load(self.prior_path), strict=True)
        for param in prior.parameters():
            param.requires_grad = False
        prior.eval()
        for i in range(self.num_agents):
            agents[i].load_state_dict(torch.load(self.prior_path), strict=True)
            agents[i].eval()

        for step in tqdm(range(self.n_steps)):
            for i in range(self.num_agents):
                samples, seqs, _ = sample_SMILES(agents[i], self.voc, n_mols=self.batch_size)

                scores = get_scores(samples, mode=self.oracle)
                samples, scores, seqs = self._memory_update(samples, scores, seqs)
            
                prior_likelihood = likelihood(prior, seqs)
                agent_likelihood = likelihood(agents[i], seqs)
                loss = torch.pow(self.sigma1 * (1 - step / self.n_steps) * to_tensor(np.array(scores)) - (prior_likelihood - agent_likelihood), 2)
                for j in range(i):
                    agent_j_likelihood = likelihood(agents[j], seqs)
                    loss -= self.sigma2 * torch.abs(agent_j_likelihood - agent_likelihood) * to_tensor(np.array(scores))
                loss = loss.mean()

                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

            self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory["scores"])), step)
            self.writer.add_scalar('top-1', self.memory["scores"][0], step)
            self.writer.add_scalar('top-10', np.mean(np.array(self.memory["scores"][:10])), step)
            self.writer.add_scalar('top-100', np.mean(np.array(self.memory["scores"][:100])), step)


            if (step + 1) % 20 == 0:
                # torch.save(agents[0].state_dict(), args.output_dir + f"QED_finetuned_{step+1}.pt")
                self.writer.add_scalar('top-100-div', int_div(list(self.memory["smiles"][:100])), step)
                self.writer.add_scalar('memory-div', int_div(list(self.memory["smiles"])), step)

            if (step + 1) % 50 == 0:
                self.memory.to_csv(f'outputs/{self.oracle}/{self.num_agents}agents+{step+1}steps.csv')

        self.memory.to_csv(f'outputs/{self.oracle}/{self.num_agents}agents+{self.n_steps}steps.csv')
        print(f'top-1 score: {self.memory["scores"][0]}')
        print(f'top-10 score: {np.mean(np.array(self.memory["scores"][:10]))}')
        print(f'top-100 score: {np.mean(np.array(self.memory["scores"][:100]))}, diversity: {int_div(list(self.memory["smiles"][:100]))}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--model_type', type=str, default="gpt")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--oracle', type=str, default="JNK3")
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma1', type=float, default=100)
    parser.add_argument('--sigma2', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--replay', type=int, default=5)
    parser.add_argument('--sim_penalize', type=bool, default=False)
    parser.add_argument('--sim_thres', type=float, default=0.7)
    parser.add_argument('--prior_path', type=str, default="ckpt/your_pretrained_model.pt")
    parser.add_argument('--vocab_path', type=str, default="data/vocab.txt")
    parser.add_argument('--output_dir', type=str, default="log/")
    args = parser.parse_args()
    print(args)

    set_seed(42)

    writer = SummaryWriter(args.output_dir + f"{args.oracle}/{args.num_agents}_{args.model_type}_{args.run_name}/")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    writer.add_text("configs", str(args))

    RL_trainer = MARL_trainer(logger=writer, configs=args)
    RL_trainer.train()

    writer.close()
    