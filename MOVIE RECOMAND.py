import pandas as pd
import numpy as np
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/content/Copy of combined_data_1.txt (1).zip',
                 header=None,
                 names=['Cust_Id', 'Rating', 'Date'])

df1 = pd.read_csv(
    '/content/Copy of movie_titles (5).csv',
    encoding='ISO-8859-1',
    header=None,
    names=['Movie_id', 'Year', 'Name'],
    on_bad_lines='skip',
    engine='python'
)

print("[1/6] Extracting Movie IDs from raw structural layout...")
movie_id = None
movie_np = []
for customer in df['Cust_Id']:
    if ":" in str(customer):
        movie_id = int(str(customer).replace(":", ""))
    movie_np.append(movie_id)

df['Movie_id'] = movie_np

df.dropna(inplace=True)
df = df[df['Rating'].notnull()]
df['Movie_id'] = df['Movie_id'].astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)

print("[2/6] Building fallback metrics for new users...")
movie_stats = df.groupby('Movie_id').agg(
    mean_rating=('Rating', 'mean'),
    rating_count=('Rating', 'count')
)
popular_movies = movie_stats[movie_stats['rating_count'] > 100]\
                      .sort_values(by='mean_rating', ascending=False)\
                      .index.tolist()

print("[3/6] Downsampling dataset for algorithmic scale...")
df_sampled = df.sample(n=min(500000, len(df)), random_state=42)
known_users = set(df_sampled['Cust_Id'].unique())

reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(df_sampled[['Cust_Id', 'Movie_id', 'Rating']], reader)

print("[4/6] Optimizing model parameters via cross-validation...")
param_grid = {
    'n_factors': [50, 100],
    'lr_all': [0.005, 0.01],
    'reg_all': [0.02, 0.1]
}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(surprise_data)
best_model = gs.best_estimator['rmse']

print("[5/6] Finalizing model fit on target downsampled framework...")
trainset = surprise_data.build_full_trainset()
best_model.fit(trainset)

print("[6/6] Launching Recommendation Interactive Interface...")
sample_users = list(known_users)[:5]
print(f"Active User IDs available in training dataset: {sample_users}")

user_input = input("Enter a User ID to execute recommendation pipeline: ")

try:
    user_input = int(user_input)
except ValueError:
    pass

if user_input not in known_users:
    print(f"\nUser ID {user_input} unrecognized. Generating Global Popularity Inventory...")
    top_trending = popular_movies[:10]
    results = df1[df1['Movie_id'].isin(top_trending)].copy()
else:
    watched_movies = df[df['Cust_Id'] == user_input]['Movie_id'].values
    all_movies = df1['Movie_id'].unique()
    unseen_movies = [m for m in all_movies if m not in watched_movies]
    
    predictions = [best_model.predict(user_input, m_id) for m_id in unseen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:10]
    
    rec_ids = [int(pred.iid) for pred in top_predictions]
    rec_scores = [pred.est for pred in top_predictions]
    
    results = df1[df1['Movie_id'].isin(rec_ids)].copy()
    score_dict = dict(zip(rec_ids, rec_scores))
    results['Predicted_Score'] = results['Movie_id'].map(score_dict)
    results = results.sort_values(by='Predicted_Score', ascending=False).reset_index(drop=True)

print("\nEngine Execution Results (Top Recommendations):")
print(results.to_string())
