import plotly.graph_objs as go
import numpy as np
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.decomposition import TruncatedSVD,PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud, STOPWORDS
from utils.config_neptune import neptune_run, neptune
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_stats(scores_graph, news_vendor):
    pd.options.mode.chained_assignment = None
    #scores distribution
    df = scores_graph[['neg_s','pos_s', 'neu_s','compound_s']]

    fig = sns.displot(data=df,
                height=5,
                aspect=3,
                kind='kde',
                fill=True,
                ).set(title ='scores distribution over corpus')
    figure = fig.figure
    neptune_run[f'stats/{news_vendor}-scores-distribution-corpus'].upload(neptune.types.File.as_image(figure))
    plt.show()
    
    #trend and sesona;ity time series analysis


    fig = make_subplots(rows=4, cols=3, subplot_titles=('Observed Pos', 'Observed Neg','Observed total','Trend Pos','Trend Neg', 'Trend total','Seasonal Pos','Seasonal Neg','Seasonal total','Residual Pos','Residual Neg', 'Residual total'))
    b_date_mean = scores_graph.groupby(by='date').mean().reset_index()

    lbl = ['Positive','Negative', 'Total sentiment']

    for idx,column in enumerate(['pos_s','neg_s','compound_s']):
        res = seasonal_decompose(b_date_mean[column], period=5, model='additive', extrapolate_trend='freq')
        
        fig.add_trace(
        go.Scatter(x=np.arange(0,len(res.observed)), y=res.observed,name='{} Observed'.format(lbl[idx])),
        row=1, col=idx+1)
        
        fig.add_trace(
        go.Scatter(x=np.arange(0,len(res.trend)), y=res.trend,name='{} Trend'.format(lbl[idx])),
        row=2, col=idx+1)
        
        fig.add_trace(
        go.Scatter(x=np.arange(0,len(res.seasonal)), y=res.seasonal,name='{} Seasonal'.format(lbl[idx])),
        row=3, col=idx+1)
        
        fig.add_trace(
        go.Scatter(x=np.arange(0,len(res.resid)), y=res.resid,name='{} Residual'.format(lbl[idx])),
        row=4, col=idx+1)
                
    fig.update_layout(height=600, width=900, title_text="Decomposition Of Our Sentiments into Trend,Level,Seasonality and Residuals")

    neptune_run[f'stats/{news_vendor}-decomposition'].upload(fig)
    fig.show()

    #positive and negative scores distribution

    plt.subplot(2,1,1)
    plt.title('Selecting A Cut-Off For Most Positive/Negative articles',fontsize=19,fontweight='bold')

    ax0 = sns.kdeplot(scores_graph['neg_s'],bw_adjust=0.1)

    kde_x, kde_y = ax0.lines[0].get_data()
    ax0.fill_between(kde_x, kde_y, where=(kde_x>0.25) , 
                    interpolate=True, color='tab:blue',alpha=0.6)

    plt.annotate('Cut-Off For Most Negative articles', xy=(0.25, 0.5), xytext=(0.4, 2),
                arrowprops=dict(facecolor='red', shrink=0.05),fontsize=16,fontweight='bold')

    ax0.axvline(scores_graph['neg_s'].mean(), color='r', linestyle='--')
    ax0.axvline(scores_graph['neg_s'].median(), color='tab:orange', linestyle='-')
    plt.legend({'PDF':scores_graph['neg_s'],r'Mean: {:.2f}'.format(scores_graph['neg_s'].mean()):scores_graph['neg_s'].mean(),
                r'Median: {:.2f}'.format(scores_graph['neg_s'].median()):scores_graph['neg_s'].median()})

    plt.subplot(2,1,2)

    ax1 = sns.kdeplot(scores_graph['pos_s'],bw_adjust=0.1,color='green')

    plt.annotate('Cut-Off For Most Positive articles', xy=(0.4, 0.43), xytext=(0.4, 2),
                arrowprops=dict(facecolor='red', shrink=0.05),fontsize=16,fontweight='bold')
    kde_x, kde_y = ax1.lines[0].get_data()
    ax1.fill_between(kde_x, kde_y, where=(kde_x>0.4) , 
                    interpolate=True, color='tab:green',alpha=0.6)
    ax1.set_xlabel('Sentiment Strength',fontsize=18)


    ax1.axvline(scores_graph['pos_s'].mean(), color='r', linestyle='--')
    ax1.axvline(scores_graph['pos_s'].median(), color='tab:orange', linestyle='-')
    plt.legend({'PDF':scores_graph['pos_s'],r'Mean: {:.2f}'.format(scores_graph['pos_s'].mean()):scores_graph['pos_s'].mean(),
                r'Median: {:.2f}'.format(scores_graph['pos_s'].median()):scores_graph['pos_s'].median()})

    filepath = "selecting-cutoff.png"
    plt.savefig(filepath)
    neptune_run[f"stats/{news_vendor}-selecting-cutoff"].upload(filepath)
    plt.show()

    #most positive most negative word clouds

    pos_tresh =scores_graph['pos_s'].quantile(0.9)
    neg_tresh = scores_graph['neg_s'].quantile(0.9)

    Most_Positive = scores_graph[scores_graph['pos_s'].between(pos_tresh,1)]
    Most_Negative = scores_graph[scores_graph['neg_s'].between(neg_tresh,1)]

    stopwords_g = stopwords.words('english')
    print(f"Most Negative: {Most_Negative['title']}")
    print(f"Most Positive: {Most_Positive['title']}")

    Most_Positive['text'] = Most_Positive['text'].str.lower()
    Most_Negative['text'] = Most_Negative['text'].str.lower()

    Most_Positive_text = ' '.join(Most_Positive['text'])
    Most_Negative_text = ' '.join(Most_Negative['text'])


    pwc = WordCloud(width=600,height=400,collocations = False,background_color='white').generate(Most_Positive_text)
    nwc = WordCloud(width=600,height=400,collocations = False,background_color='white').generate(Most_Negative_text)
    p_words = list((pwc.words_).keys())
    n_words = list((nwc.words_).keys())
    intersection =list(set(n_words) & set(p_words))
    intersection =intersection+stopwords_g
    pwc = WordCloud(width=600,height=400,collocations = False,background_color='white', stopwords = intersection, max_words=30).generate(Most_Positive_text)
    nwc = WordCloud(width=600,height=400,collocations = False,background_color='white',stopwords = intersection,max_words=30).generate(Most_Negative_text)

    plt.figure(figsize=(18,16))
    plt.subplot(1,2,1)
    plt.title('Common Words Among Most Positive articles',fontsize=16,fontweight='bold')
    plt.imshow(pwc)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('Common Words Among Most Negative articles',fontsize=16,fontweight='bold')
    plt.imshow(nwc)
    plt.axis('off')
    filepath = "wordcloud.png"
    plt.savefig(filepath)
    neptune_run[f"stats/{news_vendor}-wordcloud"].upload(filepath)

    plt.show()

    #smooth graph
    scores_graph["sum_score"]= scores_graph["compound_s"]*scores_graph["num_of_sentences"]
    scores_graph['rolling_score_sum'] = scores_graph["sum_score"].rolling(window=5).sum()
    scores_graph['rolling_num_sentences'] = scores_graph["num_of_sentences"].rolling(window=5).sum()
    scores_graph['rolling_score_by_senteces']= scores_graph['rolling_score_sum']/scores_graph['rolling_num_sentences']
    plot_smooth = scores_graph.plot(x="date", y=['rolling_score_by_senteces'],
            kind="line", figsize=(15, 6), title='average sentence score by time')
    plot_smooth.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    filepath = "smooth-plot.png"
    plt.savefig(filepath)
    neptune_run[f"{news_vendor}-stats/smooth-plot"].upload(filepath)
    plt.show()