# Spotify Tracks EDA

Exploratory data analysis on 114,000 Spotify tracks, uncovering patterns in audio features, genre characteristics, and the relationship between musical properties and popularity.

## Overview

Analyzes Spotify's audio feature data to understand what makes tracks popular, how genres differ acoustically, and whether audio features can predict popularity. The focus is on storytelling through visualization rather than modeling.

## Dataset

The [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by maharshipandya contains 114,000 tracks across 114 genres with 18 audio features including danceability, energy, valence, acousticness, tempo, and popularity. The dataset is evenly balanced at ~1,000 tracks per genre.

## Key Findings

- **Audio features do not predict popularity** — correlation analysis and popularity tier comparisons show that danceability, energy, and valence have near-zero correlation with popularity. Popularity is likely driven by external factors such as artist following, playlist placement, and marketing.
- **Energy and acousticness are strongly inversely correlated (-0.74)** — acoustic tracks are consistently low energy, while electronic and metal tracks cluster at high energy with low acousticness.
- **Genre shapes audio features more than mood does** — metal tracks are consistently high energy with low variance; classical tracks are consistently low energy. In contrast, valence (musical positivity) varies widely within every genre, including the "sad" genre.
- **pop-film and k-pop are the most popular genres on average** — with average popularity scores of ~59 and ~57 respectively, significantly above most other genres.
- **Most tracks are not popular** — popularity is right-skewed with a large spike at 0, suggesting the dataset includes many obscure or unlisted tracks that Spotify has not yet scored.

## Visualizations

- Popularity distribution (histogram + KDE)
- Audio feature distributions — danceability, energy, valence, acousticness
- Correlation heatmap of all numeric features
- Top 20 genres by average popularity (horizontal bar chart)
- Energy and valence distributions by genre (boxplots)
- Danceability, energy, and valence by popularity tier (boxplots)
- 3D scatter plot of danceability, energy, and valence colored by popularity
- 3D surface plot of average popularity across danceability and energy

## Project Structure

```
eda-portfolio/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── spotify_eda.ipynb
├── visuals/
└── README.md
```

## How to Run

Open the notebook directly in Jupyter:

```bash
jupyter notebook notebooks/spotify_eda.ipynb
```
