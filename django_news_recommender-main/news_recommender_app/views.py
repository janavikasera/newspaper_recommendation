import datetime
import numpy as np
import pandas as pd
from django.shortcuts import render
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# # 下载 NLTK 数据包
# RUN python -m nltk.downloader punkt
# RUN python -m nltk.downloader wordnet
# RUN python -m nltk.downloader omw-1.4
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# init
newsapi = NewsApiClient(api_key='524a8b2b8d4c4331b52f06c9d3673d89')
news_sources = newsapi.get_sources()
source_names = [x['name'] for x in news_sources['sources']]
a_week_ago_date = (datetime.datetime.now() - datetime.timedelta(days=7)).date().strftime("%Y-%m-%d")


def home(request):
    # get click event returns
    title = request.GET.get('title')
    content = request.GET.get('content')
    # print('title: ', title)
    # print('content: ', content)

    # get user selected filter
    query_filter = request.GET.get('query')
    # get top news
    if query_filter is None:
        top_news = newsapi.get_top_headlines(category='business')
    else:
        if 'sources=' in query_filter:
            query_filter = query_filter.replace('sources=', '')
            top_news = newsapi.get_top_headlines(sources=query_filter)
        elif 'country=' in query_filter:
            query_filter = query_filter.replace('country=', '')
            top_news = newsapi.get_top_headlines(country=query_filter)
        elif 'category=' in query_filter:
            query_filter = query_filter.replace('category=', '')
            top_news = newsapi.get_top_headlines(category=query_filter)

    # if there is no user click of any news
    if title is None:
        # if there is no recommended articles, return the top news
        if 'recommended_articles' not in request.session:
           recom_news = top_news
        else:
            # or use the last recommended articles
            print('\nUse last recommended articles...')
            recom_news = request.session.get('recommended_articles')
    else:
        print('\nRecommending preferred news based on last read...')
        # get all news articles over last week
        all_news = newsapi.get_everything(
            # q=variable,
            sources='bloomberg,the-wall-street-journal,australian-financial-review',
            from_param=a_week_ago_date,
            # to='2022-12-07',
            language='en',
            # sort_by='relevancy'
        )
        if all_news['totalResults'] > 0:
            # convert to a dataframe
            df = pd.DataFrame.from_dict(all_news['articles'])
            # drop the ones where title is too short
            # df = df[df['title'].apply(lambda x: len(x.split()) > 5)].reset_index()
            # add the clicked article
            data = []
            data.insert(0, {'title': title, 'content': content})
            df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
            # preprocessing
            df_p = preprocessing(df)
            # recommender
            recom_news = content_based_recommender(df_p, row_index=0, num_similar_items=10)
            # update session
            request.session['recommended_articles'] = recom_news
        else:
            raise Exception('No results returned')

    # content to display
    t_articles = top_news['articles']
    t_index = [i for i in range(len(t_articles))]
    t_titles = [x['title'] for x in t_articles]
    t_descs = [x['description'] for x in t_articles]
    t_urls = [x['url'] for x in t_articles]
    t_authors = [x['author'] for x in t_articles]
    t_dates = [x['publishedAt'] for x in t_articles]
    t_sources = [x['source']['name'] for x in t_articles]
    t_imgurls = [x['urlToImage'] for x in t_articles]
    t_contents = [x['content'] for x in t_articles]

    r_articles = recom_news['articles']
    r_index = [i for i in range(len(r_articles))]
    r_titles = [x['title'] for x in r_articles]
    r_descs = [x['description'] for x in r_articles]
    r_urls = [x['url'] for x in r_articles]
    r_authors = [x['author'] for x in r_articles]
    r_dates = [x['publishedAt'] for x in r_articles]
    r_sources = [x['source']['name'] for x in r_articles]
    r_imgurls = [x['urlToImage'] for x in r_articles]
    r_contents = [x['content'] for x in r_articles]

    context = {
        'top_news': list(zip(t_index, t_titles, t_descs, t_urls, t_authors, t_dates, t_sources, t_imgurls, t_contents)),
        'recom_news': list(zip(r_index, r_titles, r_descs, r_urls, r_authors, r_dates, r_sources, r_imgurls, r_contents)),
        'sources': source_names
    }

    return render(request, "index.html", context)


def preprocessing(articles):
    # remove stop words
    stop_words = set(stopwords.words('english'))
    for i in range(len(articles["content"])):
        string = ""
        for word in articles["content"][i].split():
            word = ("".join(e for e in word if e.isalnum()))
            # word = word.lower()
            if not word in stop_words:
                string += word + " "
        if i % 1000 == 0:
            print(i)  # To track number of records processed
        articles.at[i, "content"] = string.strip()
    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    for i in range(len(articles["content"])):
        string = ""
        for w in word_tokenize(articles["content"][i]):
            string += lemmatizer.lemmatize(w, pos="v") + " "
        articles.at[i, "content"] = string.strip()
        if i % 1000 == 0:
            print(i)  # To track number of records processed
    return articles


def content_based_recommender(df_article, row_index, num_similar_items):
    print('Running content_based_recommender..')
    # using a TF-IDF model
    tfidf_headline_vectorizer = TfidfVectorizer(min_df=0)
    tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(df_article['content'])
    couple_dist = pairwise_distances(tfidf_headline_features, tfidf_headline_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    # drop the index of 0 (original article)
    indices = indices[indices != 0]
    # put results in dataframe
    df_result = pd.DataFrame({'title': df_article['title'][indices].values,
                              'description': df_article['description'][indices].values,
                              'url': df_article['url'][indices].values,
                              'author': df_article['author'][indices].values,
                              'publishedAt': df_article['publishedAt'][indices].values,
                              'source': df_article['source'][indices].values,
                              'content': df_article['content'][indices].values,
                              'urlToImage': df_article['urlToImage'][indices].values,
                              'similarity': couple_dist[indices].ravel()})
    # drop 0 similarity
    df_result = df_result[df_result['similarity'] > 0].sort_values(by='similarity', ascending=False)
    print("=" * 30, "Queried article details", "=" * 30)
    print('title : ', df_article['title'][indices[0]])
    print("\n", "=" * 25, "Recommended articles : ", "=" * 23)
    print(df_result)
    # convert dataframe to dict
    results = {'articles': df_result.to_dict(orient='records')}
    return results
