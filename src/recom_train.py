import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.sparse import csr_matrix
from tqdm import tqdm


ratings_small = pd.read_csv('./data/ratings_small.csv')
ratings_small['userId'] = ratings_small['userId'].astype('category').cat.codes
ratings_small['movieId'] = ratings_small['movieId'].astype('category').cat.codes

min_rating = ratings_small['rating'].min()
max_rating = ratings_small['rating'].max()
ratings_small['rating'] = (ratings_small['rating'] - min_rating) / (max_rating - min_rating)

sparse_utility_matrix = csr_matrix(
    (ratings_small['rating'], (ratings_small['userId'], ratings_small['movieId']))
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
user_ids, movie_ids = sparse_utility_matrix.nonzero()
ratings = sparse_utility_matrix.data


max_length = min(len(user_ids), len(movie_ids), len(ratings))
user_ids = user_ids[:max_length]
movie_ids = movie_ids[:max_length]
ratings = ratings[:max_length]

users_tensor = torch.tensor(user_ids, dtype=torch.long).to(device)
movies_tensor = torch.tensor(movie_ids, dtype=torch.long).to(device)
ratings_tensor = torch.tensor(ratings, dtype=torch.float32).to(device)

class SparseDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

dataset = SparseDataset(users_tensor, movies_tensor, ratings_tensor)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, user_ids, movie_ids, ratings, device, top_k=256, lambda1=0.1, lambda2=0.1):
        super(MatrixFactorization, self).__init__()
        self.device = device
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.global_mean = ratings.mean().item()

        self.user_avg = torch.zeros(num_users, dtype=torch.float32, device=device)
        self.movie_avg = torch.zeros(num_items, dtype=torch.float32, device=device)

        self.user_count = torch.zeros(num_users, dtype=torch.float32, device=device)
        self.movie_count = torch.zeros(num_items, dtype=torch.float32, device=device)

        self.user_avg.index_add_(0, user_ids, ratings_tensor[:len(user_ids)])
        self.user_count.index_add_(0, user_ids, torch.ones_like(ratings_tensor[:len(user_ids)]))

        self.movie_avg.index_add_(0, movie_ids, ratings_tensor[:len(movie_ids)])
        self.movie_count.index_add_(0, movie_ids, torch.ones_like(ratings_tensor[:len(movie_ids)]))

        self.user_avg = self.user_avg / torch.clamp(self.user_count, min=1)
        self.movie_avg = self.movie_avg / torch.clamp(self.movie_count, min=1)

        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        self.similarity = nn.Parameter(torch.zeros(num_items, num_items, device=device))  # Full similarity matrix

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.activation = nn.ReLU()

    def compute_similarity(self, item_latent):
        similarity_full = torch.mm(item_latent, item_latent.T)
        return similarity_full

    def knn(self, item_idx, similarity_matrix, residual_matrix, top_k):
        similarity_weights = similarity_matrix[item_idx]
        top_k_values, top_k_indices = torch.topk(similarity_weights, top_k)
        top_k_residuals = residual_matrix[:, top_k_indices]
        weighted_residual = torch.sum(top_k_values * top_k_residuals.sum(dim=0))
        return weighted_residual

    def forward(self, user_indices, item_indices, sparse_utility_matrix):
        user_bias = self.user_avg[user_indices] - self.global_mean
        item_bias = self.movie_avg[item_indices] - self.global_mean
        baseline = self.global_mean + user_bias + item_bias

        user_latent = self.user_embedding(user_indices)
        item_latent = self.item_embedding(item_indices)

        similarity_matrix = self.compute_similarity(self.item_embedding.weight)

        row_indices, col_indices = sparse_utility_matrix.nonzero()
        max_length = min(len(row_indices), len(col_indices))
        row_indices = row_indices[:max_length]
        col_indices = col_indices[:max_length]
        residual_values = torch.tensor(
            sparse_utility_matrix.data[:max_length], dtype=torch.float32, device=self.device
        ) - (
            self.global_mean + self.user_avg[row_indices] + self.movie_avg[col_indices]
        )

        residual_matrix = torch.sparse_coo_tensor(
            indices = torch.tensor(np.array([row_indices, col_indices]), dtype=torch.long, device=self.device),
            values=residual_values,
            size=sparse_utility_matrix.shape,
        ).to_dense()

        residual_contrib = torch.zeros(len(item_indices), device=self.device)
        for i, item_idx in enumerate(item_indices):
            residual_contrib[i] = self.knn(item_idx, similarity_matrix, residual_matrix, self.top_k)

        svd_score = torch.sum(user_latent * item_latent, dim=1)
        predictions = self.activation(baseline + svd_score + residual_contrib)

        # Add regularization term to the loss function
        user_regularization = self.lambda1 * torch.sum(user_latent ** 2)
        item_regularization = self.lambda2 * torch.sum(item_latent ** 2)

        return predictions, user_regularization + item_regularization

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        return torch.sqrt(mse)  

num_users = users_tensor.max().item() + 1
num_items = movies_tensor.max().item() + 1
latent_dim = 30

model = MatrixFactorization(
    num_users, num_items, latent_dim, users_tensor, movies_tensor, ratings_tensor, device
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

patience = 20
best_val_loss = float('inf')
epochs_without_improvement = 0

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    total_batches = 0
    model.train()
    flag = 0
    for batch_users, batch_movies, batch_ratings in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        batch_users = batch_users.to(device)
        batch_movies = batch_movies.to(device)
        batch_ratings = batch_ratings.to(device)

        optimizer.zero_grad()
        predictions, regularization_loss = model(batch_users, batch_movies, sparse_utility_matrix)

        if epoch % 3 == 0 and flag ==0 : 
            print(f"Predictions: {predictions[:5]}")  
            print(f"Ratings:{batch_ratings[:5]}")
            flag = 1

        loss = criterion(predictions, batch_ratings) + regularization_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    model.eval()
    total_val_loss = 0
    total_val_batches = 0
    with torch.no_grad():
        for batch_users, batch_movies, batch_ratings in tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False):
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_ratings = batch_ratings.to(device)

            predictions, regularization_loss = model(batch_users, batch_movies, sparse_utility_matrix)
            loss = criterion(predictions, batch_ratings) + regularization_loss

            total_val_loss += loss.item()
            total_val_batches += 1

    avg_val_loss = total_val_loss / total_val_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), './model/CF_Model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

model.eval()
total_test_loss = 0
total_test_batches = 0
with torch.no_grad():
    for batch_users, batch_movies, batch_ratings in tqdm(test_dataloader, desc="Testing", leave=False):
        batch_users = batch_users.to(device)
        batch_movies = batch_movies.to(device)
        batch_ratings = batch_ratings.to(device)

        predictions, regularization_loss = model(batch_users, batch_movies, sparse_utility_matrix)
        loss = criterion(predictions, batch_ratings) + regularization_loss

        total_test_loss += loss.item()
        total_test_batches += 1

avg_test_loss = total_test_loss / total_test_batches
print(f"Test Loss: {avg_test_loss:.4f}")
