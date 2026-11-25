# 🎬 Movie Recommendation System: Find Films with Similar Plots

**Final project · Applied AI with Hugging Face · Topic: Movie Recommendation System**

**Team:** Nurgaliyev Batyrkhan, _Name 2_, _Name 3_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/movie-recommender/blob/main/notebook/movie_recommender.ipynb)

## Project description

The user gives a movie title (or describes a movie in their own words), and the system recommends movies with the **most similar plot**.
A pre-trained Hugging Face sentence-embedding model turns every plot description into a vector; similar plots get similar vectors,
so "find similar movies" becomes "find the closest vectors" (cosine similarity).

We then improve the recommender with a **hybrid score** that also uses actors, director and keywords, and we evaluate both versions
against genres and against **real user ratings** (MovieLens 32M).

**Research questions**
1. Can a pre-trained text model recommend relevant movies using only the plot description?
2. Does adding actors, director and keywords make the recommendations better for real users?

## Dataset

| | |
|---|---|
| **Main dataset** | [AiresPucrs/tmdb-5000-movies](https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies) on Hugging Face: a copy of the TMDB 5000 Movie Dataset ([Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), originally from [TMDB](https://www.themoviedb.org/)) |
| **Size** | 4 803 movies (4 798 after cleaning) |
| **Used columns** | `overview` (plot description, the model input), `genres` (evaluation only), `cast`, `crew`, `keywords` (hybrid model), `vote_average`, `vote_count`, `release_date` |
| **Evaluation dataset** | [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) by GroupLens: 32 million ratings from about 200 000 users |

Both datasets are downloaded automatically by the notebook. They are **not** included in this repository (their licenses do not allow redistribution).

## Model

**[`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)**: a pre-trained sentence-embedding model
(≈ 110 M parameters, 768-dimensional vectors) trained on more than 1 billion sentence pairs for semantic similarity.
We use it as it is, without fine-tuning. It is downloaded automatically (≈ 420 MB) the first time the notebook runs.

## Method

1. **Plot-only recommender:** embed every overview with MPNet → cosine similarity → top-k most similar movies.
2. **Hybrid recommender:** `score = α · plot similarity + (1 − α) · metadata similarity`, where metadata similarity is the average Jaccard overlap of the top-5 actors, the director and the keywords.
3. **Interactive search** in Colab: by movie title, by free-text description, and with the hybrid model.

## Evaluation

The dataset has no "truly similar movies" labels, so we use task-specific checks:

- **Genre check:** the model never sees the genres. Genre Precision@10 = share of the 10 recommendations that share at least one genre with the query movie.
- **Real users (MovieLens 32M):** for users with ≥ 200 liked movies (4.0+ stars), we take up to 5 liked movies as queries and count how many of the 10 recommendations appear anywhere in the same user's liked movies (Precision@10). α is chosen on 2000 **validation** users; final numbers are reported on 2000 separate **test** users.
- **Sanity checks:** popularity and rating of the recommended movies; and, among recommended movies that users actually watched, how often they rated them badly (≤ 2.5 stars).
- **Qualitative analysis** of representative examples (*The Dark Knight*, *Toy Story*, *The Notebook*, *The Conjuring*, *Interstellar*, *Fight Club*).

## Results

| Test | Random | Plot only (MPNet) | Hybrid (α = 0.2) |
|---|---|---|---|
| Genre Precision@10 (all movies) | 50.1% | **77.3%** | – |
| MovieLens Precision@10 (2000 test users) | 6.6% | 11.3% | **18.2%** |
| Watched recommendations rated ≤ 2.5★ | 16.7% | 11.8% | **9.6%** |
| Median popularity of recommendations (TMDB votes) | 254 (catalog) | 256 | 313 |

**Key findings**
- Without seeing genres, the plot model matches them far more often than random (77.3% vs 50.1%).
- Real users like plot-only recommendations 1.7× more often than random ones; the hybrid raises this to 2.8× (+61% over plot only).
- **Trade-off:** with a low α the hybrid mostly recommends films by the same director (e.g. only Christopher Nolan films for *Interstellar*), so accuracy comes at the cost of diversity.
- The popularity effect is small: hybrid recommendations are only moderately more popular than plot-only ones.

Figures and tables are in [`results/`](results/).

## Repository structure

```
movie-recommender/
├── README.md
├── requirements.txt
├── notebook/
│   └── movie_recommender.ipynb     # full project with all outputs
├── results/
│   ├── figures/                    # EDA charts, genre test, MovieLens test
│   ├── evaluation.csv              # final metrics
│   ├── alpha_validation.csv        # validation score for every α
│   ├── popularity_check.csv
│   ├── rated_check.csv
│   └── example_recommendations.csv # representative model outputs
└── report.pdf                      # summary of the project
```

## How to run

**Option A: Google Colab (recommended)**
1. Click the **Open in Colab** badge above (or upload `notebook/movie_recommender.ipynb` to Colab).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. The first cell installs the missing libraries; the datasets and the model are downloaded automatically.

Run time: about 5 minutes with a GPU, about 15–20 minutes on CPU only.

**Option B: your own computer**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/movie_recommender.ipynb
```
For an NVIDIA GPU, first install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/); the default `pip install torch` on Windows is CPU-only.
The Colab search boxes (`#@param`) become normal variables outside Colab: change the value in the code and run the cell.

## Reproducibility notes and limitations

- All random steps (user sample, validation/test split, random baseline) use a fixed seed (42).
- Results can differ by about ±0.1 percentage points between environments (CPU vs GPU, library versions); the conclusions stay the same.
- The MovieLens test uses only very active users (≥ 200 liked movies, about 4% of all users), so it shows how the system works for movie enthusiasts.
- Overviews are short (median ~50 words), so different films can sound alike, and similar films can be described differently.
- Free Colab GPUs are not always available; on CPU, encoding the movies takes several minutes.
- On some Windows setups, Python cannot verify the MovieLens download certificate; the notebook therefore downloads with the `certifi` certificate list.

## Use of AI assistants

We used Claude (Anthropic) to help structure and explain the code and to debug errors. All experiments were run, checked and interpreted by our team.
