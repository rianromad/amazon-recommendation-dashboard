#streamlit library
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from PIL import Image
import re

#visualization library
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

#data manipulation library
import pandas as pd
import numpy as np

#load model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

#css file
with open('style.css')as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

with st.sidebar:
    amazon = Image.open(r'amazon-png-logo-vector-6701.png')
    st.image(amazon, width=250)
    st.markdown('\n')
    selected = option_menu("Main Menu", ["Dashboard", 'Application'], 
        icons=['clipboard-data', 'gear'], menu_icon="cast", default_index=0)

#load data
@st.cache_data()
def load_data(url):
    df = pd.read_excel(url)
    return df

df = load_data('amazon_clean.xlsx')

if selected=="Dashboard":
    st.header('Product Recommendation Dashboard')
    with st.container():
        st.markdown('')
    m1,dum1, m2,dum2, m3, dum3  = st.columns([2,1,2,1,4,4])
    with m1:
        number_of_users = len({ i for i in df['user_id']})
        st.metric(label="Users", value=number_of_users)
    with m2:
        number_of_df = len({ i for i in df['product_name']})
        st.metric(label="Products", value=number_of_df)
    with m3:
        total_sales = "₹"+ f"{int(df['discounted_price'].sum()):,}"
        st.metric(label="Total Sales", value=total_sales)

    
    cat_opt = st.radio("Grouped chart by:",['Category','Sub-Category', 'Product'], horizontal=True)#,label_visibility="hidden")
    if cat_opt=="Category":
        viz1, viz2 = st.columns([1,1])
        with viz1:
            main_category_counts = df['main_category'].value_counts()[:10].sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                                    y=main_category_counts.index,
                                    x=main_category_counts.values,
                                    orientation='h',
                                    marker_color="#e8962f"
                                    ))
            fig.update_layout(title={
                                        'text': "Top 10 Best Seller Products",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff')
            fig.update_xaxes(title='Count')
            fig.update_yaxes(title='Category')
            #fig.update_xaxes(visible=False)
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
        with viz2:
            main_category_rating =df.groupby(['main_category'])['rating'].mean().round(2).sort_values(ascending=True).head(10)
            fig = go.Figure(go.Bar(
                                    y=main_category_rating.index,
                                    x=main_category_rating.values,
                                    orientation='h',
                                    marker_color="#e8962f"))
            fig.update_layout(title={
                                        'text': "Top 10 Products with Highest Rating",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff')
            fig.update_xaxes(title='Rating')
            fig.update_yaxes(title='Category')
            #fig.update_xaxes(visible=False)
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
    elif cat_opt=="Sub-Category":
        viz1, viz2 = st.columns([1,1])
        with viz1:
            sub_category_counts = df['sub_category'].value_counts()[:10].sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                                    y=sub_category_counts.index,
                                    x=sub_category_counts.values,
                                    orientation='h',
                                    marker_color="#e8962f"
            ))
            fig.update_layout(title={
                                        'text': "Top 10 Best Seller Products",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff')
            fig.update_xaxes(title='Count')
            fig.update_yaxes(title='Sub-Category')
            #fig.update_xaxes(visible=False)
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
        with viz2:
            sub_category_rating =df.groupby(['sub_category'])['rating'].mean().round(2).sort_values(ascending=True).head(10)
            fig = go.Figure(go.Bar(
                                    y=sub_category_rating.index,
                                    x=sub_category_rating.values,
                                    orientation='h',
                                    marker_color="#e8962f"))
            fig.update_layout(title={
                                        'text': "Top 10 Products with Highest Rating",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff')
            #fig.update_xaxes(visible=False)
            fig.update_xaxes(title='Rating')
            fig.update_yaxes(title='Sub-Category')
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
    else:
        viz1, viz2 = st.columns([1,1])
        with viz1:
            product_counts = df['product_name'].value_counts()[:10].sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                                    y=product_counts.index,
                                    x=product_counts.values,
                                    orientation='h',
                                    marker_color="#e8962f",
                                    
            ))
            fig.update_layout(title={
                                        'text': "Top 10 Best Seller Products",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff', hovermode='y')
            fig.update_xaxes(title='Count')
            fig.update_yaxes(title='Product Name', automargin=True)
            #fig.update_yaxes(visible=False)
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
        with viz2:
            product_rating =df.groupby(['product_name'])['rating'].mean().round(2).sort_values(ascending=True).head(10)
            fig = go.Figure(go.Bar(
                                    y=product_rating.index,
                                    x=product_rating.values,
                                    orientation='h',
                                    marker_color="#e8962f"
                                    ))
            fig.update_layout(title={
                                        'text': "Top 10 Products with Highest Rating",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff', hovermode='y')
            #fig.update_yaxes(visible=False)
            fig.update_xaxes(title='Rating')
            fig.update_yaxes(title='Product Name', automargin=True)
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
    
    cslid, slid,dum4 = st.columns([3,2,4])
    with cslid:
        rate_opt = st.radio("Rating Options:",['Greater than','Less than', 'Equal'], horizontal=True)#,label_visibility="hidden")
    with slid:
        rate_slid = st.slider("Rating:", min_value=3.0,max_value=5.0,value=3.0,step=0.1)

    wc,hist_rating = st.columns([4,3])

    with wc:
        norm=plt.Normalize(0,700)
        cmap = colors.LinearSegmentedColormap.from_list("", ["#FCCB8F","#e8962f"])
        if rate_opt=="Greater than":
            selected = df[df['rating'] >= rate_slid]
        elif rate_opt=="Less than":
            selected = df[df['rating'] <= rate_slid]
        else:
            selected = df[df['rating'] == rate_slid]

        # reviews_text = ' '.join(selected['review_content'].dropna().values().strip().lower())
        stopwords = ['n','व','वस','ह','च','त','प','द','ल','1','2','3','4','5','6','7','8','9','0','.',',','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        reviews_text =[i.lower() for i in  re.split(r'\W+',' '.join(selected['review_content'].dropna().values))]
        reviews_text = [i for i in reviews_text if i not in stopwords]

        def word_count(words):
            counts = dict()
            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1

            return counts
        
        wc = pd.Series(word_count(reviews_text)).sort_values(ascending=False).head(25)
        fig = go.Figure(go.Bar(
                                    y=wc.values,
                                    x=wc.index,
                                    orientation='v',
                                    marker_color="#e8962f"))
        fig.update_layout(title={
                                    'text': "Word Distribution",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},plot_bgcolor='#ffffff')
        #fig.update_yaxes(visible=False)
        fig.update_xaxes(title='Word')
        fig.update_yaxes(title='Count', automargin=True)
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)
        
    with hist_rating:
        fig = px.histogram(df, x="rating",
                    title='Rating Distribution',
                    labels={'Harga':'Harga Sewa (Rp)'}, # can specify one label per df column
                    opacity=0.8,
                    color_discrete_sequence=['#e8962f'],
                    )
        fig.update_layout(title={
                    'y':0.9,
                    'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},plot_bgcolor='#ffffff', showlegend=False)
        fig.update_xaxes(title='Rating')
        fig.update_yaxes(title='Count')
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)

    user, cm = st.columns([1,1])
    with user:
        user_counts = df['user_id_encoded'].astype('object').value_counts().sort_values(ascending=False)[:10].sort_values(ascending=True)
        fig = go.Figure(go.Bar(
                                y=['User '+str(i) for i in user_counts.index],
                                x=user_counts.values,
                                orientation='h',
                                marker_color="#e8962f",
                                
        ))
        fig.update_layout(title={
                                    'text': "Top 10 Most Active User",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},plot_bgcolor='#ffffff')
        fig.update_xaxes(title='Number of Transaction')
        fig.update_yaxes(title='User Id', automargin=True)
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)
    with cm:
        norm=plt.Normalize(0,1)
        cmap = colors.LinearSegmentedColormap.from_list("", ["white","#e8962f"])
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop('user_id_encoded', axis=1)
        correlation_matrix = numeric_cols.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, ax=ax)
        ax.set_title("Correlation Matrix", pad=20, fontweight='bold', fontsize=12)
        st.pyplot(fig, use_container_width=True)

elif selected=="Application":
    st.header('Product Recommendation App')


    with st.container():
        st.write('Recommendation Method:')
        method = st.radio('Recommendation Method:',['Collaborative Filtering', 'Item-Based Filtering'],horizontal=True, label_visibility='collapsed')
    
    if method == "Collaborative Filtering":
        col1, col2 = st.columns([1,4])
        with col1:
            st.markdown("#### User ID")
            user_id = st.selectbox("User ID: ", set(df["user_id_encoded"].to_list()), label_visibility='collapsed')

        st.markdown("#### Favourite Products")
        purchase_history = df[df["user_id_encoded"]==user_id].sort_values(["rating"], ascending=False)[['product_name','rating']].rename(columns={'product_name':'Product Name','rating':'Rating'}).sort_values('Product Name',ascending=True).set_index('Product Name')
        
        st.dataframe(purchase_history, use_container_width=True)

        st.markdown("#### Similar Users")
        users_matrix=df.pivot_table(index=["user_id_encoded"],columns=["main_category"],values="rating").fillna(0)
        index = df.index[df["user_id_encoded"]==user_id][0]

        try:
                similarity = cosine_similarity(users_matrix)
                similar_users = list(enumerate(similarity[index]))
                similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[0:5]
                k = [k for (k,v) in similar_users]
                v = [v for (k,v) in similar_users]
                df2 = pd.DataFrame({"User Id":k,f"Similarity Score (User ID: {user_id})":v}).set_index("User Id")
                st.dataframe(df2, use_container_width=True)
        except:
            st.error("No similar user !")

        st.markdown("#### Recommended Products")
        users_id = []
        
        for i in similar_users:
                data = df[df["user_id_encoded"]==users_matrix.index[i[0]]]
                users_id.extend(list(data.drop_duplicates("user_id_encoded")["user_id_encoded"].values))
                
        x = df[df["user_id_encoded"]==user_id]
        recommend_product = []
        user = list(users_id)

        for i in user:
                y = df[df["user_id_encoded"]==i]
                sim_items = y.loc[~y["product_name"].isin(x["product_name"]), :]
                sim_items = sim_items.sort_values(["rating"], ascending=False)[0:5]
                recommend_product.extend(sim_items["product_name"].values)

        recommend_product_df = df[df['product_name'].isin(recommend_product)][['product_name','rating']].rename(columns={'product_name':'Product Name','rating':'Rating'}).groupby('Product Name').mean().round(2).sort_values('Rating',ascending=False)
        st.dataframe(recommend_product_df, use_container_width=True)
    
    else:
        st.markdown("#### Product Category")
        category = st.selectbox("Category:",set(sorted(df['main_category'].to_list())), label_visibility='collapsed')
        
        st.markdown("#### Product Name")
        product = st.selectbox("Product:",set(sorted(df[(df['main_category']==category)]['product_name'].to_list())),label_visibility='collapsed')

        st.markdown("#### Product Recommendation")

        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        # Apply TF-IDF vectorization to 'text' column
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        index = df[df['product_name'] == product].index[0]

        # Get the pairwise similarity scores of the product
        sim_scores = list(enumerate(cosine_sim[index]))

        # Sort the products based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 similar products
        top_products = pd.DataFrame({'Product Name':[df.iloc[score[0]]['product_name'] for score in sim_scores[1:11]],
                                     'Similarity Score':[score[1] for score in sim_scores[1:11]]}).groupby('Product Name').mean().round(2).sort_values('Similarity Score',ascending=False)
        
        st.dataframe(top_products, use_container_width=True)
