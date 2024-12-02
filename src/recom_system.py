import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ratings_small = pd.read_csv('./data/ratings_small.csv')


ratings_small['userId'] = ratings_small['userId'].astype('category')
ratings_small['movieId'] = ratings_small['movieId'].astype('category')
min_rating = ratings_small['rating'].min()
max_rating = ratings_small['rating'].max()
ratings_small['rating'] = (ratings_small['rating'] - min_rating) / (max_rating - min_rating)


ratings_small['userId_cat'] = ratings_small['userId'].cat.codes
ratings_small['movieId_cat'] = ratings_small['movieId'].cat.codes


user_id_mapping = dict(enumerate(ratings_small['userId'].cat.categories))
movie_id_mapping = dict(enumerate(ratings_small['movieId'].cat.categories))


sparse_utility_matrix = csr_matrix(
    (ratings_small['rating'], (ratings_small['userId_cat'], ratings_small['movieId_cat']))
)


user_ids, movie_ids = sparse_utility_matrix.nonzero()
ratings = sparse_utility_matrix.data

user_ids = torch.tensor(user_ids, dtype=torch.long).to(device)
movie_ids = torch.tensor(movie_ids, dtype=torch.long).to(device)
ratings_tensor = torch.tensor(ratings, dtype=torch.float32).to(device)

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

        user_regularization = self.lambda1 * torch.sum(user_latent ** 2)
        item_regularization = self.lambda2 * torch.sum(item_latent ** 2)

        return predictions, user_regularization + item_regularization

def load_model(model_path, num_users, num_items, latent_dim, device, user_ids, movie_ids, ratings_tensor):
    model = MatrixFactorization(num_users, num_items, latent_dim, user_ids, movie_ids, ratings_tensor, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def predict_top_10_unrated_movies(user_id, model, sparse_utility_matrix, ratings_small, device):
    model.eval()  

    if user_id not in ratings_small['userId'].values:
        print(f"User {user_id} not found in the dataset.")
        return

    encoded_user_id = ratings_small.loc[ratings_small['userId'] == user_id, 'userId_cat'].iloc[0]

    user_rated_movies_encoded = sparse_utility_matrix[encoded_user_id].nonzero()[1]

    if len(user_rated_movies_encoded) == 0:
        print(f"User {user_id} has not rated any movies.")
        return

    # 사용자가 평가하지 않은 영화들 찾기
    all_movie_ids_encoded = torch.arange(sparse_utility_matrix.shape[1], dtype=torch.long).to(device)
    unrated_movies_encoded = list(set(all_movie_ids_encoded.tolist()) - set(user_rated_movies_encoded.tolist()))

    if len(unrated_movies_encoded) == 0:
        print(f"User {user_id} has rated all movies.")
        return

    user_tensor = torch.tensor([encoded_user_id] * len(unrated_movies_encoded), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(unrated_movies_encoded, dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_ratings, _ = model(user_tensor, movie_tensor, sparse_utility_matrix)

    predicted_ratings = predicted_ratings.cpu().numpy()

    # 예측된 평점과 영화 ID 매핑
    predicted_ratings_df = pd.DataFrame({
        'movieId': [movie_id_mapping[movie_id] for movie_id in unrated_movies_encoded],
        'predicted_rating': predicted_ratings
    })

    top_10_movies = predicted_ratings_df.sort_values(by='predicted_rating', ascending=False).head(10)

    top_10_movies['predicted_rating'] = top_10_movies['predicted_rating'] * (max_rating - min_rating) + min_rating

    print(top_10_movies)

def predict_ratings_for_rated_movies(user_id, model, sparse_utility_matrix, ratings_small, device):
    model.eval() 


    if user_id not in ratings_small['userId'].values:
        print(f"User {user_id} not found in the dataset.")
        return

    encoded_user_id = ratings_small.loc[ratings_small['userId'] == user_id, 'userId_cat'].iloc[0]

    user_rated_movies_encoded = sparse_utility_matrix[encoded_user_id].nonzero()[1]

    if len(user_rated_movies_encoded) == 0:
        print(f"User {user_id} has not rated any movies.")
        return

    user_tensor = torch.tensor([encoded_user_id] * len(user_rated_movies_encoded), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(user_rated_movies_encoded, dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_ratings, _ = model(user_tensor, movie_tensor, sparse_utility_matrix)
    predicted_ratings = predicted_ratings.cpu().numpy()

    user_rated_movies = [movie_id_mapping[movie_id] for movie_id in user_rated_movies_encoded]

    predicted_ratings = predicted_ratings * (max_rating - min_rating) + min_rating

    results = pd.DataFrame({
        'movieId': user_rated_movies,
        'predicted_rating': predicted_ratings
    })

    return results


if __name__ == "__main__":
    model_path = './model/CF_Model.pth'

    num_users = ratings_small['userId_cat'].nunique()
    num_items = ratings_small['movieId_cat'].nunique()
    latent_dim = 30 

    model = load_model(model_path, num_users, num_items, latent_dim, device, user_ids, movie_ids, ratings_tensor)
    
    user_id = int(input("추천해줄 user_id를 입력하세요: "))
    predict_top_10_unrated_movies(user_id, model, sparse_utility_matrix, ratings_small, device)
