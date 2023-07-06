#streamlit library
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from PIL import Image

#visualization library
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors
from wordcloud import WordCloud
import seaborn as sns

#data manipulation library
import pandas as pd
import numpy as np

#load model
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
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
    df = pd.read_csv(url)
    return df

products = load_data('amazon_clean.csv')

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

    # copt1, copt2, dum4 = st.columns([2,2,3])
    # with copt1:
    #     st.markdown('\n')
    #     st.markdown('\n')
    #     st.write("Grouped chart by: ")
    # with copt2:
        #cat_opt = st.radio("Group chart by:",['Category','Sub-Category'], horizontal=True)#,label_visibility="hidden")

    #major1, major2 = st.columns([2,1])
    #with major1:
    cat_opt = st.radio("Grouped chart by:",['Category','Sub-Category', 'Product'], horizontal=True)#,label_visibility="hidden")
    if cat_opt=="Category":
        viz1, viz2 = st.columns([1,1])
        with viz1:
            main_category_counts = products['main_category'].value_counts()[:10].sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                                    y=main_category_counts.index,
                                    x=main_category_counts.values,
                                    orientation='h',
                                    marker_color="#e8962f",hoverinfo='none',
                                    text = main_category_counts.values
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
                                    marker_color="#e8962f",hoverinfo='none',
                                    text =main_category_rating.values))
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
                                    marker_color="#e8962f",hoverinfo='none',
                                    text = sub_category_counts.values
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
                                    marker_color="#e8962f",hoverinfo='none',
                                    text =sub_category_rating.values))
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
                                    text = product_counts.values, hoverinfo='none'
            ))
            fig.update_layout(title={
                                        'text': "Top 10 Best Seller Products",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff')
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
                                    marker_color="#e8962f",
                                    text =product_rating.values, hoverinfo='none'))
            fig.update_layout(title={
                                        'text': "Top 10 Products with Highest Rating",
                                        'y':0.9,
                                        'x':0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'},plot_bgcolor='#ffffff')
            #fig.update_yaxes(visible=False)
            fig.update_xaxes(title='Rating')
            fig.update_yaxes(title='Product Name', automargin=True)
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            st.plotly_chart(fig,use_container_width=True)
    # with major2:
    #     st.markdown('\n')
    #     st.markdown('\n')
    #     st.markdown('\n')
    #     st.markdown('\n')
  
    #     fig = px.histogram(products, x="rating",
    #                 title='Rating Distribution',
    #                 labels={'Harga':'Harga Sewa (Rp)'}, # can specify one label per df column
    #                 opacity=0.8,
    #                 color_discrete_sequence=['#e8962f'],
    #                 )
    #     fig.update_layout(title={
    #                 'y':0.9,
    #                 'x':0.5,
    #                     'xanchor': 'center',
    #                     'yanchor': 'top'},plot_bgcolor='#ffffff', showlegend=False)
    #     fig.update_xaxes(title='Rating')
    #     fig.update_yaxes(title='Count')
    #     fig.layout.xaxis.fixedrange = True
    #     fig.layout.yaxis.fixedrange = True
    #     st.plotly_chart(fig,use_container_width=True)
    
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

        reviews_text = ' '.join(selected['review_content'].dropna().values).strip().lower()
        wordcloud = WordCloud(background_color='white', min_font_size=10,colormap=cmap).generate(reviews_text)
        fig, ax = plt.subplots(facecolor=None)
        ax.imshow(wordcloud)
        ax.set_axis_off()
        ax.set_title("Wordcloud Distribution", pad=15, fontweight='bold', fontsize=11)
        fig.tight_layout()
        st.pyplot(fig,use_container_width=True)
    
    with hist_rating:
        # norm=plt.Normalize(0,1)
        # cmap = colors.LinearSegmentedColormap.from_list("", ["white","#e8962f"])
        # numeric_cols = products.select_dtypes(include=['float64', 'int64']).drop('Unnamed: 0', axis=1)
        # correlation_matrix = numeric_cols.corr()
        # fig, ax = plt.subplots()
        # sns.heatmap(correlation_matrix, annot=True, cmap=cmap, ax=ax)
        # ax.set_title("Correlation Matrix", pad=20, fontweight='bold', fontsize=15)
        # st.pyplot(fig, use_container_width=True)

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
                                text = user_counts.values, hoverinfo='none'
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
        numeric_cols = products.select_dtypes(include=['float64', 'int64']).drop('Unnamed: 0', axis=1)
        correlation_matrix = numeric_cols.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, ax=ax)
        ax.set_title("Correlation Matrix", pad=20, fontweight='bold', fontsize=12)
        st.pyplot(fig, use_container_width=True)

elif selected=="Application":
   st.header('Product Recommendation Application')
  
