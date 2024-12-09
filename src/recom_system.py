import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# GPU 또는 CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
ratings_small = pd.read_csv('./data/ratings_small.csv')
movies = pd.read_csv("./data/movies_metadata.csv", low_memory=False)
keywords = pd.read_csv("./data/keywords.csv")

# 데이터 전처리
ratings_small['userId'] = ratings_small['userId'].astype('category')
ratings_small['movieId'] = ratings_small['movieId'].astype('category')
ratings_small['rating'] = (ratings_small['rating'] - ratings_small['rating'].min()) / (ratings_small['rating'].max() - ratings_small['rating'].min())

ratings_small['userId_cat'] = ratings_small['userId'].cat.codes
ratings_small['movieId_cat'] = ratings_small['movieId'].cat.codes

# 매핑 생성
user_id_mapping = dict(enumerate(ratings_small['userId'].cat.categories))
movie_id_mapping = dict(enumerate(ratings_small['movieId'].cat.categories))

# 유틸리티 행렬 생성
sparse_utility_matrix = csr_matrix(
    (ratings_small['rating'], (ratings_small['userId_cat'], ratings_small['movieId_cat']))
)

# Tensor로 변환
user_ids, movie_ids = sparse_utility_matrix.nonzero()
ratings = sparse_utility_matrix.data

user_ids = torch.tensor(user_ids, dtype=torch.long).to(device)
movie_ids = torch.tensor(movie_ids, dtype=torch.long).to(device)
ratings_tensor = torch.tensor(ratings, dtype=torch.float32).to(device)


# 사용자 프로필 생성 함수
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'

def build_user_profile(user_id):
    user_ratings = ratings_small[ratings_small['userId'] == user_id].copy()
    if user_ratings.empty:
        return {}

    user_ratings['time_of_day'] = pd.to_datetime(user_ratings['timestamp'], unit='s').dt.hour.apply(time_of_day)
    user_rated_movies = user_ratings.merge(movies[['id', 'content']], left_on='movieId', right_on='id', how='inner')
    if user_rated_movies.empty:
        return {}

    user_profile = {}
    for time, group in user_rated_movies.groupby('time_of_day'):
        # 가중치를 적용하여 문자열을 여러 번 반복
        group['weighted_content'] = group.apply(lambda row: (row['content'] + ' ') * int(row['rating']), axis=1)
        user_profile[time] = ' '.join(group['weighted_content'])
    return user_profile



class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, user_ids, movie_ids, ratings, device, top_k=256, lambda1=0.1, lambda2=0.1):
        super(MatrixFactorization, self).__init__()
        self.device = device
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # 전체 평균
        self.global_mean = ratings.mean().item()

        # 사용자와 아이템 평균 계산
        self.user_avg = torch.zeros(num_users, dtype=torch.float32, device=device)
        self.movie_avg = torch.zeros(num_items, dtype=torch.float32, device=device)

        self.user_count = torch.zeros(num_users, dtype=torch.float32, device=device)
        self.movie_count = torch.zeros(num_items, dtype=torch.float32, device=device)

        # 각 사용자와 아이템에 대해 평점 더하기
        self.user_avg.index_add_(0, user_ids, ratings_tensor[:len(user_ids)])
        self.user_count.index_add_(0, user_ids, torch.ones_like(ratings_tensor[:len(user_ids)]))

        self.movie_avg.index_add_(0, movie_ids, ratings_tensor[:len(movie_ids)])
        self.movie_count.index_add_(0, movie_ids, torch.ones_like(ratings_tensor[:len(movie_ids)]))

        # 사용자와 아이템의 평균 계산 (분모가 0인 경우를 대비하여)
        self.user_avg = self.user_avg / torch.clamp(self.user_count, min=1)
        self.movie_avg = self.movie_avg / torch.clamp(self.movie_count, min=1)

        # Latent embeddings
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # K-NN 기반 유사도 행렬 초기화
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

# 모델 로드 함수
def load_model(model_path, num_users, num_items, latent_dim, device, user_ids, movie_ids, ratings_tensor):
    model = MatrixFactorization(num_users, num_items, latent_dim, user_ids, movie_ids, ratings_tensor, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def hybrid_recommendations(user_id, model, sparse_utility_matrix, ratings_small, movies, tfidf, tfidf_matrix, top_n=10):
    model.eval()
    if user_id not in ratings_small['userId'].values:
        print(f"User {user_id} not found in the dataset.")
        return pd.DataFrame({'Title': [], 'Hybrid Score': []})

    # Collaborative Filtering 예측
    encoded_user_id = ratings_small.loc[ratings_small['userId'] == user_id, 'userId_cat'].iloc[0]
    all_movie_ids_encoded = torch.arange(sparse_utility_matrix.shape[1], dtype=torch.long).to(device)
    user_rated_movies_encoded = sparse_utility_matrix[encoded_user_id].nonzero()[1]
    unrated_movies_encoded = list(set(all_movie_ids_encoded.tolist()) - set(user_rated_movies_encoded.tolist()))

    user_tensor = torch.tensor([encoded_user_id] * len(unrated_movies_encoded), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(unrated_movies_encoded, dtype=torch.long).to(device)

    with torch.no_grad():
        collaborative_ratings, _ = model(user_tensor, movie_tensor, sparse_utility_matrix)
        collaborative_ratings = collaborative_ratings * 4 + 1  # 스케일을 1~5로 변환

    # 사용자 프로필 생성 및 시간대 계산
    user_profile = build_user_profile(user_id)
    if not user_profile:
        print("No recommendations available for this user.")
        return pd.DataFrame({'Title': [], 'Hybrid Score': []})

    # 시간대 정보 포함된 콘텐츠 데이터 생성
    for time, profile_content in user_profile.items():
        # 시간대를 콘텐츠에 명시적으로 추가
        movies[f'content_{time}'] = movies['content'] + f' time:{time}'

    # 전체 콘텐츠 벡터화
    all_contents = movies[[f'content_{time}' for time in user_profile.keys()]].fillna('').apply(' '.join, axis=1)
    all_tfidf_matrix = tfidf.fit_transform(all_contents)

    # 사용자 시간대 프로필 벡터화 및 콘텐츠 유사도 계산
    content_scores = {}
    for time, profile_content in user_profile.items():
        user_profile_vector = tfidf.transform([profile_content + f' time:{time}'])
        similarity = cosine_similarity(user_profile_vector, all_tfidf_matrix)
        movies['similarity'] = similarity.flatten()

        for movie_id, score in zip(movies['id'], movies['similarity']):
            if movie_id not in content_scores:
                content_scores[movie_id] = 0
            content_scores[movie_id] += score

    # Content-Based 점수 스케일링 (0~1 -> 1~5)
    max_content_score = max(content_scores.values(), default=0)
    min_content_score = min(content_scores.values(), default=0)
    for movie_id in content_scores:
        content_scores[movie_id] = 4 * (content_scores[movie_id] - min_content_score) / (max_content_score - min_content_score + 1e-6) + 1

    # Hybrid 점수 계산
    hybrid_scores = []
    for movie_id, collab_score in zip(unrated_movies_encoded, collaborative_ratings.cpu().numpy()):
        original_movie_id = movie_id_mapping[movie_id]
        content_score = content_scores.get(original_movie_id, 1)  # 기본값 1 (스케일 범위 하한)
        hybrid_score = 0.5 * collab_score + 0.5 * content_score
        hybrid_scores.append((original_movie_id, hybrid_score))

    # 결과 정렬 및 반환
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]
    recommendations = pd.DataFrame(hybrid_scores, columns=['movieId', 'Hybrid Score'])
    recommendations = recommendations.merge(movies[['id', 'original_title']], left_on='movieId', right_on='id')
    recommendations = recommendations.drop_duplicates(subset='movieId')

    return recommendations[['original_title', 'Hybrid Score']].sort_values(by='Hybrid Score', ascending=False)


# 실행
if __name__ == "__main__":
    model_path = '/home/jyjy0425/dm/best_matrix_factorization_model.pth'
    num_users = ratings_small['userId_cat'].nunique()
    num_items = ratings_small['movieId_cat'].nunique()
    latent_dim = 30

    # 모델 불러오기
    model = load_model(model_path, num_users, num_items, latent_dim, device, user_ids, movie_ids, ratings_tensor)
    user_id = 1
    # 영화 데이터 전처리
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    ratings_small['movieId'] = pd.to_numeric(ratings_small['movieId'], errors='coerce')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')

    common_ids = set(ratings_small['movieId']).intersection(set(movies['id']))
    ratings_small = ratings_small[ratings_small['movieId'].isin(common_ids)]
    movies = movies[movies['id'].isin(common_ids)]

    movies['genres'] = movies['genres'].fillna('[]').apply(eval).apply(
        lambda x: [d['name'] for d in x] if isinstance(x, list) else []
    )
    movies = movies.merge(keywords, on='id', how='left')
    movies['keywords'] = movies['keywords'].fillna('[]').apply(eval).apply(
        lambda x: [d['name'] for d in x] if isinstance(x, list) else []
    )
    movies['content'] = movies['genres'] + movies['keywords']
    movies['content'] = movies['content'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

    # TF-IDF 생성
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    recommendations = hybrid_recommendations(user_id=user_id, model=model, 
                                         sparse_utility_matrix=sparse_utility_matrix, 
                                         ratings_small=ratings_small, 
                                         movies=movies, 
                                         tfidf=tfidf,  
                                         tfidf_matrix=tfidf_matrix, 
                                         top_n=20)

if not recommendations.empty:
    print(recommendations.to_string(index=False, header=["Title", "Hybrid Score"], justify="center"))
else:
    print("No recommendations available.")
