# Annals Explorer

Annals Explorer is a Streamlit-based digital humanities prototype for
exploring selected entries from the **Annals of Ulster (600--700 CE)**.

## Features

-   Search the Annals using a chatbot
-   Explore historical people and places
-   View source evidence linked to each record
-   Visualise relationships using interactive knowledge graphs
-   Explore NLP- and data science-derived historical information

## Project Structure

``` text
Annal/
├── Website.py
├── requirements.txt
├── packages.txt
├── README.md
├── annals_events_U600_U700.csv
├── website_people.csv
├── website_places.csv
├── website_events.csv
├── website_edges.csv
└── website_summary.csv
```

## Install

``` bash
pip install -r requirements.txt
```

## Run

``` bash
streamlit run Website.py
```

## Deployment

Deploy directly from this GitHub repository using Streamlit Community
Cloud.

## Developed By

**Dr. Sudhansu Bala Das**

Postdoctoral Researcher, University of Galway

Insight Research Ireland Centre for Data Analytics

## Academic Supervision

**Prof. Pádraic Moran**

Classics & Celtic Studies, University of Galway
