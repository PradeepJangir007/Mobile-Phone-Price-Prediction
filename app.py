import streamlit as st
import pandas as pd
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind

import my_module
from my_module import SparseToDenseTransformer,LogStandardScaler,confidence_
sys.modules['__main__'].SparseToDenseTransformer = SparseToDenseTransformer
sys.modules['__main__'].confidence_ = confidence_

df=pickle.load(open('df.pkl','rb'))
model=pickle.load(open('smart_phone_price_with_PLSR.pkl','rb'))

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["📈 Price Distribution" , "📊 Analysis",  "📱 Price Prediction"])
if page == "📱 Price Prediction":
    st.title('📱 Smart Phone Price Prediction')
    st.subheader('Fill in the smartphone specifications:')
    st.markdown('**Note:** Yes = 1 , No = 0')

    # brand
    Brand=st.selectbox('Brand',df.company.unique())

    model_series=st.selectbox('Model_series',df[df['company']==Brand].Model_series.unique())
    Dual_sim=st.selectbox('Dual sim',['1','0'])
    VoLTE=st.selectbox('VoLTE',['1','0'])
    G=st.selectbox('5G',['1','0'])
    Vo5G=st.selectbox('Vo5G',['1','0'])
    Foldable_Display=st.selectbox('Foldable Display',['1','0'])
    Dual_Displayd=st.selectbox('Dual Display',['1','0'])

    Processor=st.selectbox('Processor',df.Processor_.unique())
    disply=st.text_input('Display resulation', '2400*1080')
    size=st.text_input('Display size', '6.5')
    Ram = st.text_input('RAM (in GB)', '8')
    Processor_series = st.text_input('Processor Series', '730')
    External_Memory = st.text_input('External Memory (GB)', '512')
    Inbuilt_memory = st.text_input('Inbuilt Memory (GB)', '128')
    fast_charging = st.text_input('Fast Charging (Watt)', '25')
    Battery = st.text_input('Battery Capacity (mAh)', '5000')
    No_of_Rear = st.text_input('Number of Rear Cameras', '3')
    No_of_Front = st.text_input('Number of Front Cameras', '1')
    Primary_rear_camera = st.text_input('Primary Rear Camera (MP)', '50')
    Primary_front_camera = st.text_input('Primary Front Camera (MP)', '16')
    Number_of_core = st.text_input('Number of Cores', '8')
    b=st.button('predict price')
    if b==True:
        v=int(str(disply).split('*')[0])
        h=int(str(disply).split('*')[1])
        pixal_D=(v**2+h**2)**0.5
        PPI=((v**2+h**2)/float(size))**0.5
        Processor_series=my_module.Processor_s.Transformer(Processor,Processor_series)
        x={'Model_series':[model_series],'Dual Sim':[int(Dual_sim)],'VoLTE':[int(VoLTE)],'5G': [int(G)],
            'Vo5G':[int(Vo5G)],'Ram':[int(Ram)],'Battery':[int(Battery)],'Foldable Display':[int(Foldable_Display)],
            'Dual Display':[int(Dual_Displayd)],'External_Memory':[int(External_Memory)],
            'company':[Brand],'Inbuilt_memory':[int(Inbuilt_memory)],'fast_charging':[float(fast_charging)],
            'Processor_':[Processor],'Processor_series':[int(Processor_series)],'No _of_Rear':No_of_Rear,'No _of_Front':[int(No_of_Front)],
            'Primary_rear_camera':[Primary_rear_camera],'Primary_front_camera':[Primary_front_camera],
            'Number_of_core':[Number_of_core],
            'PPI':[PPI]}
        x=pd.DataFrame(x)
        st.title('price ' +str(round(2.7183**((model.predict(x))[0]))))
        conf=confidence_()
        L=conf.interval(x)[0]
        U=conf.interval(x)[1]
        st.subheader(f'confidence interval with 90% \n {(round(2.7183 ** L), round(2.7183 ** U))}')
# --------------------- PAGE 2: ANALYSIS ---------------------
elif page == "📊 Analysis":
    st.title("📊 Smartphone Data Analysis")
    st.write("This section is under development.")
    st.info("💡 You can visualize trends, compare brands, or explore feature importance here.")
    data = pd.read_csv('smart_phone_data.csv')  # Example data file
    # Example placeholder for future visualization
    if st.checkbox("Show sample data"):
        st.dataframe(data.head())
    # Sidebar - Company and Feature Selection
    st.sidebar.header("📌 Filters")
    companies = ['None'] + sorted(data['company'].unique().tolist())
    selected_company = st.sidebar.selectbox("Select Company", companies)

    # Only columns appropriate for categorical X-axis
    possible_features = data.columns.tolist()
    exclude_features = [
        'Price', 'company', 'pixal_D', 'PPI', 'No _of_Rear',
        'Battery','Model_series','No _of_Front'
    ]
    categorical_or_numeric = [col for col in possible_features if col not in exclude_features]

    selected_x = st.sidebar.selectbox("Select Feature (X-axis)", categorical_or_numeric)

    if selected_company != 'None':
        # Filter data
        filtered_data = data[data['company'] == selected_company]
    else:
        filtered_data = data

    if not filtered_data.empty:
        st.subheader(f"🎯 Price vs **{selected_x}** for {selected_company}")
        
        # Create two columns
        col1, col2 = st.columns(2)

        # Left Column – Boxplot
        with col1:
            st.markdown("📦 **Boxplot: Price Distribution**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=filtered_data, x=selected_x, y='Price', ax=ax1)
            ax1.set_xlabel(selected_x)
            ax1.set_ylabel("Price (₹)")
            st.pyplot(fig1)

        # Right Column – Barplot (Mean Prices)
        with col2:
            st.markdown("📊 **Barplot: Mean Price**")
            mean_prices = filtered_data.groupby(selected_x)['Price'].mean().reset_index()

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=mean_prices, x=selected_x, y='Price', ax=ax2)

            # Add value labels on top of each bar
            for index, row in mean_prices.iterrows():
                ax2.text(index, row['Price'] + 0.01 * mean_prices['Price'].max(),  # y = slightly above bar
                        f"{row['Price']:.0f}", ha='center', va='bottom', fontsize=9)

            ax2.set_xlabel(selected_x)
            ax2.set_ylabel("Mean Price (₹)")
            st.pyplot(fig2)

        # Get number of groups
        groups = filtered_data[selected_x].nunique()

        # Run analysis based on number of groups
        if groups > 2:
            st.subheader("📊 ANOVA Table (Price ~ " + selected_x + ")")
            try:
                model = ols(f'Price ~ C({selected_x})', data=filtered_data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(anova_table)
                p_val = anova_table['PR(>F)'][0]
                if p_val < 0.05:
                    st.success("✅ Statistically significant difference (p < 0.05)")
                else:
                    st.info("ℹ️ No significant difference (p ≥ 0.05)")
            except Exception as e:
                st.error(f"Error in ANOVA: {e}")

        elif groups == 2:
            st.subheader("📏 T-Test Between Two Groups")

            try:
                group_labels = filtered_data[selected_x].unique()
                group1 = filtered_data[filtered_data[selected_x] == group_labels[0]]['Price']
                group2 = filtered_data[filtered_data[selected_x] == group_labels[1]]['Price']

                t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                st.markdown(f"**Groups**: {group_labels[0]} vs {group_labels[1]}")
                st.markdown(f"**T-statistic**: {t_stat:.4f}")
                st.markdown(f"**P-value**: {p_val:.4f}")
                if p_val < 0.05:
                    st.success("✅ Statistically significant difference (p < 0.05)")
                else:
                    st.info("ℹ️ No significant difference (p ≥ 0.05)")
            except Exception as e:
                st.error(f"Error in T-Test: {e}")

        else:
            st.warning("Not enough groups in the selected feature for statistical test.")

    else:
        st.warning("No data available for the selected company.")
elif page == "📈 Price Distribution":
    st.title("📈 Smartphone Price Distribution")

    # Load data
    data = pd.read_csv("smart_phone_data.csv")

    # Sidebar filters
    st.sidebar.header("📌 Filters")

    # 1. Company selection
    companies = ['All'] + sorted(data['company'].dropna().unique().tolist())
    selected_company = st.sidebar.selectbox("Select Company", companies)

    # Base filtering
    if selected_company == 'All':
        filtered_data = data.copy()
        st.subheader("📊 Price Distribution for All Companies")
    else:
        filtered_data = data[data['company'] == selected_company]
        st.subheader(f"📊 Price Distribution for {selected_company}")

    # 2. Optional second filter (categorical column + value)
    final_data = filtered_data.copy()  # start with already filtered

    if selected_company != 'All' and not filtered_data.empty:
        possible_features = data.columns.tolist()
        exclude_features = [
            'Price', 'company', 'pixal_D', 'PPI', 'No _of_Rear',
            'Battery','Model_series','No _of_Front','company', 'Price'
        ]
        categorical_or_numeric = [col for col in possible_features if col not in exclude_features]
    
        if categorical_or_numeric:
            selected_col = st.sidebar.selectbox("Further Filter By (Optional)", ['All'] + categorical_or_numeric)

            if selected_col != 'All':
                unique_vals = ['All'] + sorted(filtered_data[selected_col].dropna().unique().tolist())
                selected_value = st.sidebar.selectbox(f"Select value for '{selected_col}'", unique_vals)

                if selected_value != 'All':
                    final_data = filtered_data[filtered_data[selected_col] == selected_value]
                    #st.subheader(f"📊 Price Distribution for {selected_company} - {selected_col} = {selected_value}")
                else:
                    final_data = filtered_data.copy()  # reset to company-level only
            else:
                final_data = filtered_data.copy()  # reset to company-level only
    else:
        final_data = filtered_data.copy()

    # Plotting section
    if final_data.empty:
        st.warning("No data available for selected filters.")
    else:
        col1, col2 = st.columns(2)

        # Histogram with KDE
        with col1:
            st.markdown("📉 **Histogram with KDE**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(final_data['Price'], kde=True, bins=20, ax=ax1, color='skyblue')
            ax1.set_xlabel("Price (₹)")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        # Horizontal Boxplot
        with col2:
            st.markdown("📦 Boxplot")
            fig2, ax2 = plt.subplots(figsize=(6, 3.7))
            sns.boxplot(x=final_data['Price'], ax=ax2, )
            ax2.set_xlabel("Price (₹)")
            st.pyplot(fig2)

        # Summary statistics
        st.markdown("📌 **Summary Statistics**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📦 Count", f"{final_data['Price'].count():.0f}")
        col2.metric("💰 Mean", f"₹{final_data['Price'].mean():.0f}")
        col3.metric("📉 Min", f"₹{final_data['Price'].min():.0f}")
        col4.metric("📈 Max", f"₹{final_data['Price'].max():.0f}")
        col5.metric("📊 Std Dev", f"{final_data['Price'].std():.0f}")


else:
    st.info("Please select a company to begin analysis.")

    

