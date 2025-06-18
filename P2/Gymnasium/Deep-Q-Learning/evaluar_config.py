import argparse
import json
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class AgenteDQLearning:
    def __init__(self, env, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 alpha=1e-3, gamma=0.99, buffer_size=100000, batch_size=64,
                 target_update_freq=1000):
        
        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.target_update_freq = target_update_freq

        self.nS = env.observation_space.shape[0]
        self.nA = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.nS, self.nA).to(self.device)
        self.target_net = DQN(self.nS, self.nA).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.stats = []
        self.episode_lengths = []
        self.steps_done = 0

    def guardar_agente(self, name="agente_dqn.pkl"):
        # Crear un diccionario con todo lo importante
        modelo_guardado = {
            "params": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "buffer_size": self.buffer.maxlen,
                "batch_size": self.batch_size,
                "target_update_freq": self.target_update_freq,
            },
            "policy_state_dict": self.policy_net.state_dict(),
            "stats": self.stats,
            "episode_lengths": self.episode_lengths,
        }
        
        # Guardar en un archivo
        with open(name, "wb") as f:
            pickle.dump(modelo_guardado, f)

    @staticmethod
    def cargar_agente(name="agente_dqn.pkl"):
        # Cargar el archivo
        with open(name, "rb") as f:
            data = pickle.load(f)
        
        # Reconstruir el entorno
        env = gym.make("MountainCar-v0")
        env.reset(seed=SEED)
        
        # Reconstruir el agente con los mismos hiperparámetros
        agent = AgenteDQLearning(env, **data["params"])
        agent.policy_net.load_state_dict(data["policy_state_dict"])
        agent.stats = data["stats"]
        agent.episode_lengths = data["episode_lengths"]
    
        return agent


    def seleccionar_accion(self, estado):
        """Selecciona una acción usando la política epsilon-soft explícita."""
        policy = self._epsilon_soft_policy(estado)
        return np.random.choice(np.arange(self.nA), p=policy)

    def seleccionar_accion_greedy(self, estado):
        """Selecciona la mejor acción según la política actual (greedy)."""
        estado_tensor = torch.tensor(estado, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(estado_tensor)
        return torch.argmax(q_vals, dim=1).item()

    
    def _epsilon_soft_policy(self, estado):
        """Devuelve una política epsilon-soft como vector de probabilidades."""
        estado_tensor = torch.tensor(estado, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(estado_tensor).cpu().numpy().squeeze()
        
        policy = np.ones(self.nA) * self.epsilon / self.nA
        best_action = np.argmax(q_vals)
        policy[best_action] += 1.0 - self.epsilon
        return policy


    def almacenar(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def _actualizar_red(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(a), dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1).to(self.device)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32).to(self.device)
        d = torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(1).to(self.device)

        q_vals = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_net(s_).max(1, keepdim=True)[0]
            target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def entrenar(self, num_episodes=5000, mostrar_barra=True):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.env.reset(seed=SEED)
        self.env.action_space.seed(SEED)

        acumulador_recompensas = 0.0

        pbar = tqdm(range(num_episodes), desc="Entrenando", dynamic_ncols=True)
        
        for t in pbar:
            estado, _ = self.env.reset()
            done = False
            total_reward = 0
            pasos = 0

            while not done:
                accion = self.seleccionar_accion(estado)
                estado_siguiente, recompensa, terminado, truncado, _ = self.env.step(accion)
                done = terminado or truncado

                self.almacenar(estado, accion, recompensa, estado_siguiente, done)
                self._actualizar_red()

                estado = estado_siguiente
                total_reward += recompensa
                pasos += 1
                self.steps_done += 1

            self.episode_lengths.append(pasos)
            acumulador_recompensas += total_reward
            self.stats.append(acumulador_recompensas / (t + 1))

            pbar.set_postfix({'Recompensa': total_reward})
            
            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return self.policy_net



def evaluar_configuracion_dqn(alpha, gamma, epsilon, episodes):
    env = gym.make("MountainCar-v0")
    agente = AgenteDQLearning(env,
                              alpha=alpha,
                              gamma=gamma,
                              epsilon=epsilon,
                              epsilon_decay=0.995,
                              epsilon_min=0.01,
                              buffer_size=100000,
                              batch_size=64,
                              target_update_freq=1000)
    
    agente.entrenar(num_episodes=episodes, mostrar_barra=False)
    recompensa_final = np.mean(agente.stats[-100:])
    return (alpha, gamma, epsilon, recompensa_final)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--env", type=str, default="MountainCar-v0")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    resultado = evaluar_configuracion_dqn(args.alpha, args.gamma, args.epsilon, args.episodes)
    
    with open(f"resultados/resultado_{args.id}.json", "w") as f:
        json.dump(resultado, f, indent=2)
