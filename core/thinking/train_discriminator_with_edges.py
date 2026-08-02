# core/thiking/train_discriminator_with_edges.py

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from generate_hypothesis import KnowledgeGraph, cosine_sim

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

class DiscriminatorWithEdges(nn.Module):
    def __init__(self, input_dim=384*2 + 1 + 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, task_emb, combo_emb, edge_features):
        dot = (task_emb * combo_emb).sum(dim=1, keepdim=True)
        combined = torch.cat([task_emb, combo_emb, dot, edge_features], dim=1)
        return self.net(combined)

def train_discriminator_with_edges(dataset_file, epochs=200, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    kg = KnowledgeGraph()
    kg.load_from_db(embedder)

    X_task, X_combo, X_edge, y = [], [], [], []
    for item in data:
        task_emb = embedder.encode(item['task'])
        task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
        node_ids = item['combination']
        # Извлекаем плоский список узлов (аналогично prepare_dataset_with_edges)
        if isinstance(node_ids, list) and len(node_ids) > 0:
            if isinstance(node_ids[0], list):
                node_ids = [nid for sublist in node_ids for nid in sublist]
        elif isinstance(node_ids, dict):
            node_ids = list(node_ids.values())
        node_ids = [str(nid) for nid in node_ids]

        # Фильтруем только те узлы, которые есть в графе
        valid_nodes = [nid for nid in node_ids if kg.get_node_embedding(nid) is not None]
        if len(valid_nodes) < 2:
            continue

        combo_emb = np.mean([kg.get_node_embedding(nid) for nid in valid_nodes], axis=0)
        combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
        edge_feat = item['edge_features']
        edge_vec = np.array([edge_feat['edge_sim_mean'], edge_feat['edge_density'], edge_feat['type_diversity']])
        X_task.append(task_emb)
        X_combo.append(combo_emb)
        X_edge.append(edge_vec)
        y.append(item['score'])

    if len(X_task) == 0:
        print("❌ Нет валидных примеров для обучения.")
        return

    X_task_t = torch.FloatTensor(np.array(X_task)).to(device)
    X_combo_t = torch.FloatTensor(np.array(X_combo)).to(device)
    X_edge_t = torch.FloatTensor(np.array(X_edge)).to(device)
    y_t = torch.FloatTensor(np.array(y)).unsqueeze(1).to(device)

    dataset = TensorDataset(X_task_t, X_combo_t, X_edge_t, y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    discriminator = DiscriminatorWithEdges().to(device)
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for task_emb, combo_emb, edge_feat, score in dataloader:
            pred = discriminator(task_emb, combo_emb, edge_feat)
            loss = loss_fn(pred, score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    torch.save(discriminator.state_dict(), "discriminator_with_edges.pth")
    print("✅ Дискриминатор с учётом рёбер сохранён.")

if __name__ == "__main__":
    train_discriminator_with_edges('training_data_inventions_with_edges.json', epochs=500)




# import json
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sentence_transformers import SentenceTransformer
# from generate_hypothesis import KnowledgeGraph, cosine_sim
#
# DB_CONFIG = {
#     "host": "localhost",
#     "database": "postgres",
#     "user": "postgres",
#     "password": "postgres"
# }
#
# class DiscriminatorWithEdges(nn.Module):
#     def __init__(self, input_dim=384*2 + 1 + 3):  # task_emb + combo_emb + dot + 3 edge features
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, task_emb, combo_emb, edge_features):
#         dot = (task_emb * combo_emb).sum(dim=1, keepdim=True)
#         combined = torch.cat([task_emb, combo_emb, dot, edge_features], dim=1)
#         return self.net(combined)
#
# def train_discriminator_with_edges(dataset_file, epochs=200, batch_size=32):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     with open(dataset_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     # Загружаем эмбеддинги узлов
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
#     kg = KnowledgeGraph()
#     kg.load_from_db(embedder)
#
#     X_task, X_combo, X_edge, y = [], [], [], []
#     for item in data:
#         task_emb = embedder.encode(item['task'])
#         task_emb = task_emb / (np.linalg.norm(task_emb) + 1e-8)
#         node_ids = item['combination']
#         combo_emb = np.mean([kg.get_node_embedding(nid) for nid in node_ids], axis=0)
#         combo_emb = combo_emb / (np.linalg.norm(combo_emb) + 1e-8)
#         edge_feat = item['edge_features']
#         edge_vec = np.array([edge_feat['edge_sim_mean'], edge_feat['edge_density'], edge_feat['type_diversity']])
#         X_task.append(task_emb)
#         X_combo.append(combo_emb)
#         X_edge.append(edge_vec)
#         y.append(item['score'])
#
#     # Преобразуем в тензоры
#     X_task_t = torch.FloatTensor(np.array(X_task)).to(device)
#     X_combo_t = torch.FloatTensor(np.array(X_combo)).to(device)
#     X_edge_t = torch.FloatTensor(np.array(X_edge)).to(device)
#     y_t = torch.FloatTensor(np.array(y)).unsqueeze(1).to(device)
#
#     dataset = TensorDataset(X_task_t, X_combo_t, X_edge_t, y_t)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     discriminator = DiscriminatorWithEdges().to(device)
#     optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()
#
#     for epoch in range(epochs):
#         for task_emb, combo_emb, edge_feat, score in dataloader:
#             pred = discriminator(task_emb, combo_emb, edge_feat)
#             loss = loss_fn(pred, score)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: loss={loss.item():.4f}")
#
#     torch.save(discriminator.state_dict(), "discriminator_with_edges.pth")
#     print("✅ Дискриминатор с учётом рёбер сохранён.")
#
# if __name__ == "__main__":
#     train_discriminator_with_edges('training_data_inventions_with_edges.json', epochs=500)