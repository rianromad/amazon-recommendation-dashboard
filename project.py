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

products = load_data('amazon_clean.xlsx')

if selected=="Dashboard":
    st.header('Product Recommendation Dashboard')
    m1,dum1, m2,dum2, m3, dum3  = st.columns([2,1,2,1,4,4])
    with m1:
        number_of_users = len({ i for i in products['user_id']})
        st.metric(label="Users", value=number_of_users)
    with m2:
        number_of_products = len({ i for i in products['product_name']})
        st.metric(label="Products", value=number_of_products)
    with m3:
        total_sales = "₹"+ f"{int(products['discounted_price'].sum()):,}"
        st.metric(label="Total Sales", value=total_sales)

    
    cat_opt = st.radio("Grouped chart by:",['Category','Sub-Category', 'Product'], horizontal=True)#,label_visibility="hidden")
    if cat_opt=="Category":
        viz1, viz2 = st.columns([1,1])
        with viz1:
            main_category_counts = products['main_category'].value_counts()[:10].sort_values(ascending=True)
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
            main_category_rating =products.groupby(['main_category'])['rating'].mean().round(2).sort_values(ascending=True).head(10)
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
            sub_category_counts = products['sub_category'].value_counts()[:10].sort_values(ascending=True)
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
            sub_category_rating =products.groupby(['sub_category'])['rating'].mean().round(2).sort_values(ascending=True).head(10)
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
            product_counts = products['product_name'].value_counts()[:10].sort_values(ascending=True)
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
            product_rating =products.groupby(['product_name'])['rating'].mean().round(2).sort_values(ascending=True).head(10)
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
    
    cslid, slid,dum4 = st.columns([2,2,3])
    with cslid:
        rate_opt = st.radio("Rating Options:",['Greater than','Less than'], horizontal=True)#,label_visibility="hidden")
    with slid:
        rate_slid = st.slider("Rating:", min_value=3.0,max_value=5.0,value=3.0,step=0.1)

    wc,hist_rating = st.columns([4,3])

    with wc:
        norm=plt.Normalize(0,700)
        cmap = colors.LinearSegmentedColormap.from_list("", ["#FCCB8F","#e8962f"])
        if rate_opt=="Greater than":
            selected = products[products['rating'] >= rate_slid]
        else:
            selected = products[products['rating'] <= rate_slid]

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
        fig = px.histogram(products, x="rating",
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
        user_counts = products['user_id_encoded'].astype('object').value_counts().sort_values(ascending=False)[:10].sort_values(ascending=True)
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
        numeric_cols = products.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_cols.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, ax=ax)
        ax.set_title("Correlation Matrix", pad=20, fontweight='bold', fontsize=12)
        st.pyplot(fig, use_container_width=True)

elif selected=="Application":
   st.header('Product Recommendation Application')
   
   col1, col2 = st.columns([1,1])
   with col1:
       user_id = st.selectbox("User ID: ", set(products["user_id_encoded"].to_list()))

   st.markdown("### Favourite Products")
   purchase_history = products[products["user_id_encoded"]==user_id].sort_values(["rating"], ascending=False)[['product_name','rating']].rename(columns={'product_name':'Product Name','rating':'Rating'}).sort_values('Product Name',ascending=True).set_index('Product Name')
   #product_rating = products[products['product_name'].isin(purchase_history['product_name'])][['product_name','rating']].groupby('product_name').mean().round(2)

   #st.dataframe(product_rating, use_container_width=True)
   st.dataframe(purchase_history, use_container_width=True)

   st.markdown("### Similar Users")
   users_matrix=products.pivot_table(index=["user_id_encoded"],columns=["main_category"],values="rating").fillna(0)
   index = products.index[products["user_id_encoded"]==user_id][0]

   try:
        similarity = cosine_similarity(users_matrix)
        similar_users = list(enumerate(similarity[index]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[0:5]
        k = [k for (k,v) in similar_users]
        v = [v for (k,v) in similar_users]
        df = pd.DataFrame({"User Id":k,f"Similarity Score (User ID: {user_id})":v}).set_index("User Id")
        st.dataframe(df, use_container_width=True)
   except:
       st.error("No similar user !")

   st.markdown("### Recommended Products")
   users_id = []
   
   for i in similar_users:
        data = products[products["user_id_encoded"]==users_matrix.index[i[0]]]
        users_id.extend(list(data.drop_duplicates("user_id_encoded")["user_id_encoded"].values))
        
   x = products[products["user_id_encoded"]==user_id]
   recommend_products = []
   user = list(users_id)

   for i in user:
        y = products[products["user_id_encoded"]==i]
        sim_items = y.loc[~y["product_name"].isin(x["product_name"]), :]
        sim_items = sim_items.sort_values(["rating"], ascending=False)[0:5]
        recommend_products.extend(sim_items["product_name"].values)

   recommend_products_df = products[products['product_name'].isin(recommend_products)][['product_name','rating']].rename(columns={'product_name':'Product Name','rating':'Rating'}).groupby('Product Name').mean().round(2)
   st.dataframe(recommend_products_df, use_container_width=True)