import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime

st.set_page_config(page_title="Wikipedia Political Interest Analysis",layout="wide")





#ALL TABS
main, data_sum, features, classification, hypothesis, graphs, summary = st.tabs(["1.0 Introduction",
        "2.0 Data Summary",
        "3.0 New Features",
        "4.0 Text Classification",
        "5.0 Hypothesis Testing",
        "6.0 Interactive Visualization",
        "7.0 Summary"
        ])

# --- Load Data ---
df = pd.read_csv("daily_label_summary.csv")
sample = pd.read_csv("sample.csv")

c_codes = {
    "USA": 'US',
    "UK": "GB",
    "Canada": "CS",
    "Australia": "AU",
    "India": "IN"
}


df['date'] = pd.to_datetime(df['date'])







with main:
    st.title("How do Wikipedia pageviews of political topics vary across countries during the 2023–2024 period, particularly during electoral seasons?")
    st.markdown("---")

    st.markdown("""
### Research Questions

- ###### How do Wikipedia pageviews of political topics vary across countries during the 2023–2024 period, particularly during electoral seasons?
- ###### Are certain countries more likely to engage with political content than others, and do we see spikes in attention that correspond to major political events?
""")

    st.markdown("In order to measure political engagement with political related articles on WIkipedia I used the pageviews of the top 10,000 most viewed articles per country in 2023-2024. Using these I classified the articles as political or non-political based on each unique article's description in Wikidata. Once all articles were classified I was able to group how many political articles were viewed during different date ranges in different countries and analyze increases or spikes in engagement. I will determine spikes in political ppageviews as increased political engagement and will try to determine specific political events or elections in those date ranges.")
    st.markdown("""
I chose to focus on 5 countries:

- **Australia**  
- **Canada**  
- **India**  
- **United States**  
- **United Kingdom**
""")

    st.markdown("Using the **English Wikipedia** within **2023-02-06 to 2023-02-12**")
    st.markdown("The reason I chose these countries is because they all had varying types of elections during 2023 and 2024")

    data = {
        "Country": ["United States", "United Kingdom", "Australia", "Canada", "India"],
        "Election Type": [
            "General Election + Primaries",
            "General Election",
            "State/Territory Elections",
            "By-elections (Federal election due 2025)",
            "General Election (Lok Sabha)"
        ],
        "Date Range": [
            "Primaries Jan–Jun 2024; General Nov 5, 2024",
            "Campaign May–Jun 2024; Election Jul 4, 2024",
            "NSW Mar 25, 2023; NT Aug 24, 2024; QLD Oct 26, 2024",
            "Various by-elections in 2023–2024",
            "Apr 19 – Jun 1, 2024 (7 phases); Results Jun 4, 2024"
        ]
    }

    df_elections = pd.DataFrame(data)

    st.header("Major Election Date Ranges (2023–2024)")
    st.table(df_elections)
    
    st.markdown("""
#### What I expect to find out
I expect to see a larger percentage of the political articles amongst the pageviews of different countries during important elections.
I also expect to see an increase in the slope for political pageviews depending on how important an election is (e.g. General vs State elections)

#### Hypotheses
- Countries undergoing national elections in 2023–2024 will show a significant increase in political article views during election months compared to non-election months.
- Non-political articles will dominate overall traffic, but political articles will show sharper temporal fluctuations.

#### Findings and Key Takeaways
- **Spikes tend to align with major political events:** Spikes in political pageviews tend to align with election or result days, leadership changes, protests, or other political events. This helps support the idea engagement with political topics in Wikipedia increases during major political events such as elections.
- **Largest spikes occur during general elections**
- After a large spike there tend to be smaller spikes that follow which insinuate continued interest in these political events
- **November 6** tends to show large spikes across all English speaking countries analyzed (US General Election)

#### Limitations and Ethical Considerations
- Due to the graphs consisting only of pageviews data I am unable to determine the exact article that cause spikes in political pageviews. This led to outside investigation and inferences on what could've caused these spikes
- Topics seen as political can be subjective (e.g. Historical events, Geographical locations, Religion, etc.) and false positives can still occur
- Countries tend to use Wikipedia differently, not all countries actively use Wikipedia so the amount of pageviews in different countries display a large contrast and may not directly be tied to how engaged a country is to politics.
- Not everyone has equal access to the internet, Wikipedia, English-language content,  or digital literacy. The dataset only reflects those who can access English Wikipedia and not all people who care about politics.
- Event bias: High pageviews don't reflect importance, they may just reflect media coverage and higher controversies
- Spikes happen when an event may be recent or have current media coverage but don't reflect long-term political engagement
""")






with data_sum:
    st.header("Data Summary")
    st.markdown("The dataset consists of the **pageviewes of the top 10,000 most-viewed Wikipedia articles** across five countries (United States, United Kingdom, Canada, Australia, and India) during the years **2023–2024**.")
    st.markdown("**Time Interval:** February 2023 – December 2024")
    st.markdown("**Wiki:** English Wikipedia (`en.wikipedia`)")
    st.markdown("**Countries:** United States, United Kingdom, Canada, Australia, India")

    st.markdown("**CSV File:** The CSV files consists of the 'date', 'country_code', 'label', and total 'pageviews' of that label and country for that day")
    # --- Dataset Preview ---
    st.header("Dataset Overview")
    country = st.selectbox("Select a country to preview its dataset:", c_codes.keys())
    st.write("#### Sample Rows")
    st.dataframe(df.head())
    st.write("#### Summary Statistics")
    st.write(df.describe())




with features:
    st.header("New Features")
    st.markdown("""
#### get_wikidata_description(qid, retries=5)
Using the QID columns from the DPDP files, I queiried Wiidata to retrieve unique article descriptions which helped give context to each article (describing a person, event, place, or concept)

#### classify_article(text, vectorizer, classifier)
Using the descriptions from the articles I would classify the article using a Naive Bayes Classifier

#### Sample
Once I was able to classify each unique article, I added a new column to the csv file with each article's label
The table below displays a sample of the new column added to the Australia CSV file:              
""")
    st.dataframe(sample.head())

    st.markdown("""
### Final CSV FIle
Once each article was labeled in the 5 CSV files, I made a smaller CSV file which contained the sums of political, non-political, and No QID pageviews per country.
""")
    st.write("#### Sample Rows")
    st.dataframe(df.head())







with classification:
    st.header("Poltical vs Non-Political: Naive Bayes Classifier")
    st.markdown("""
### Classification for Identifying politic related articles

#### Getting the text
**1.** Accumulated text of the first paragraphs of **political Wikipedia articles** based on the elections listed in https://en.wikipedia.org/wiki/List_of_elections_in_2024 and https://en.wikipedia.org/wiki/List_of_elections_in_2023, focusing on the following countries: USA, Canada, UK, Australia, and India. Added paragraphs from articles in the politics section of https://wikimediafoundation.org/news/2024/12/03/announcing-english-wikipedias-most-popular-articles-of-2024/since there are several articles listed in relation to politics

**2.** Accumulating text of the first paragraphs of **non-political Wikipedia articles** based on the top wikipedia articles listed in https://wikimediafoundation.org/news/2023/12/05/announcing-wikipedias-most-popular-articles-of-2023/ and https://wikimediafoundation.org/news/2024/12/03/announcing-english-wikipedias-most-popular-articles-of-2024/

#### Creating the Classifier
**3.** Combining "political" amd "non-political" and cleaning up the text

**4.** Veztorizing the text and using MultinomialNB

#### Testing the Classifier
**5.** Tested the classifier with some testing paragraphs and descriptions: All tests came out accurate

#### Evaluate Classifier
**Accuracy:** 92.795  
**F1 Score:** 92.795  
""")




with hypothesis:
    st.header("Hypothesis Testing: Country Engagement with Political Topics")

    st.markdown("""
### Resarch Questions(s)

- ###### How do Wikipedia pageviews of political topics vary across countries during the 2023–2024 period, particularly during electoral seasons?
- ###### Are certain countries more likely to engage with political content than others, and do we see spikes in attention that correspond to major political events?
                
### Hypothesis
- Countries undergoing national elections in 2023–2024 will show a significant increase in political article views during election months compared to non-election months.
- Non-political articles will dominate overall traffic, but political articles will show sharper temporal fluctuations.

### Measurement
To test our hypothesis we are visually comparing changes in political pageviews since the sample is too big and the p-value will not make sense.        
- Shifts and spikes in political pageviews serve as a proxy for interest in political topics
- Higher percentages of portion of political pageviews serve as a proxy for more engagement in that country
- Determining if spikes in pageviews align with elections or other political events

### Hypothesis Testing   
For all five countries together, we can test:

**Null Hypothesis (H₀):**  
All countries have the **same mean proportion** of political-page views with little to no shifts. 

**Alternative Hypothesis (H₁):**  
At least one country differs.  

""")



with graphs:
    st.header("Interactive Visualizations")
    st.markdown("""
- AU: Australia
- CA: Canada
- GB: United Kingdom
- IN: India
- US: United States
""")
    # --- Bar Chart: Political vs Non-Political ---
    st.header("Political vs Non-Political Pageviews")

    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    selected_years = st.slider(
        "Select date range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        step=timedelta(days=30)
    )

    min_selected, max_selected = selected_years
    mask = (df['date'].dt.date >= min_selected) & (df['date'].dt.date <= max_selected)
    filtered_df = df[mask]

    bar_data = (
        filtered_df
        .groupby(["country_code", "label"])["views"]
        .sum()
        .reset_index()
    )

    selected_country = st.multiselect(
        "Countries to display:",
        options=bar_data['country_code'].unique(),
        default=bar_data['country_code'].unique()
    )

    bar_data  = bar_data[bar_data['country_code'].isin(selected_country)]

    fig_bar = px.bar(
        bar_data,
        x="country_code",
        y="views",
        color="label",
        barmode="group",
        title="Political vs Non-Political Pageviews"
    )

    st.plotly_chart(fig_bar, use_container_width=True)
   
    # --- Line Chart: Political Articles Over Time ---
    st.header("Political Article Trends Over Time")

    min_date, max_date = df['date'].min().date(), df['date'].max().date()

    selected_years = st.slider(
        "Select date range of Wikipedia pageviews to display:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        step=timedelta(days=30)
    )

    min_selected, max_selected = selected_years
    mask = (df['date'].dt.date >= min_selected) & (df['date'].dt.date <= max_selected)
    filtered_df = df[mask]

    aggregation = st.radio(
        "Show data by:",
        ["Daily", "Monthly"],
        horizontal=True
    )

    if aggregation == "Daily":
        grouped = (
            filtered_df[filtered_df['label'] == "political"]
            .groupby([pd.Grouper(key='date', freq='D'), 'country_code'])['views']
            .sum()
            .reset_index(name='political_views')
        )
    else:
        grouped = (
            filtered_df[filtered_df['label'] == "political"]
            .groupby([pd.Grouper(key='date', freq='M'), 'country_code'])['views']
            .sum()
            .reset_index(name='political_views')
        )

    selected_country = st.multiselect(
        "Countries to display:",
        options=grouped['country_code'].unique(),
        default=grouped['country_code'].unique(),
        key="lineplot"
    )

    grouped = grouped[grouped['country_code'].isin(selected_country)]

    fig_line = px.line(
        grouped,
        x="date",
        y="political_views",
        color="country_code",
        title=f"Total Political Article Views Over Time ({aggregation})"
    )

    fig_line.update_layout(
        legend_title_text="Country",
        margin=dict(t=80, b=40),
        height=500
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # --- Stacked Bar Chart: Distribution of Labels by Country ---
    st.header("Distribution of Article Types by Country")

    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    selected_range = st.slider(
        "Select date range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    start, end = selected_range

    all_labels = ["No QID", "non-political", "political"]
    selected_labels = st.multiselect(
        "Select which labels to include:",
        options=all_labels,
        default=all_labels
    )
    filtered_df = df[df["label"].isin(selected_labels)]

    mask = (filtered_df['date'].dt.date >= start) & (filtered_df['date'].dt.date <= end)
    filtered_df = filtered_df[mask]


    agg = (
        filtered_df
        .groupby(["country_code", "label"])["views"]
        .sum()
        .reset_index()
    )

    total_views = agg.groupby("country_code")["views"].transform("sum")
    agg["percent"] = agg["views"] / total_views * 100

    fig = px.bar(
        agg,
        x="country_code",
        y="views",
        color="label",
        title="Stacked Distribution of Article Pageview Types by Country",
        barmode="stack",
        hover_data={
            "views": True,
            "percent": ":.1f",
            "label": True,
            "country_code": False
        }
    )

    fig.update_layout(
        yaxis_title="Total Views",
        legend_title_text="Article Type",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Sample Table")
    st.dataframe(grouped.head(20))





with summary:
    st.header("Summary and Ethical Considerations")
    st.markdown("""
### Key Takeaways
##### Percentage of Political Pageviews
Overall percentage of political pageviews seen in all countries stayed consistent
**Percentage of political pageviews** ranged from **28 - 33%** for all countries

##### Analyzing Spikes in Political Pageviews
- **Spikes tend to align with major political events:** Spikes in political pageviews tend to align with election or result days, leadership changes, protests, or other political events. This helps support the idea engagement with political topics in Wikipedia increases during major political events such as elections.
- **Largest spikes occur during general elections**
- After a large spike there tend to be smaller spikes that follow which insinuate continued interest in these political events
- **November 6** tends to show large spikes across all English speaking countries analyzed (US General Election)
- United States displays highest absolute engagement
- Australia and Canada show smaller trends in pageviews

### Limitations
- Due to the graphs consisting only of pageviews data I am unable to determine the exact article that cause spikes in political pageviews. This led to outside investigation and inferences on what could've caused these spikes
- The use of English Wikipedia also limits the population because, although most of these countries consist of primarily English speakers, there are other other languages that should be taken into consideration before deciding on a final conclusion.
- Topics seen as political can be subjective (e.g. Historical events, Geographical locations, Religion, etc.) and false positives can still occur
- Countries tend to use Wikipedia differently, not all countries actively use Wikipedia so the amount of pageviews in different countries display a large contrast and may not directly be tied to how engaged a country is to politics.

### Ethical Considerations
- Not everyone has equal access to the internet, Wikipedia, English-language content,  or digital literacy. The dataset only reflects those who can access English Wikipedia and not all people who care about politics.
- Event bias: High pageviews don't reflect importance, they may just reflect media coverage and higher controversies
- Spikes happen when an event may be recent or have current media coverage but don't reflect long-term political engagement
""")



    us_events_df = pd.DataFrame([
        ["USA", "2023-03-15", "13.29M", "Federal Reserve decision period; national political attention on banking instability"],
        ["USA", "2023-10-25", "13.76M", "U.S. House elects new Speaker (Mike Johnson)"],
        ["USA", "2023-11-13 to 2023-11-27", "12.74M - 13.6M", "Congressional funding negotiations and national political coverage"],
        ["USA", "2024-07-15", "11.04M", "Major 2024 election-year activity; conventions and campaign events"],
        ["USA", "2024-07-22", "10.57M", "Continuation of 2024 convention season and national political events"],
        ["USA", "2024-11-06", "21.16M", "Post–2024 general election coverage (Election Day was Nov 5)"],
        ["USA", "2024-11-13", "11.09M", "Post-election transition period and certification processes"]
    ], columns=["country", "date", "pageviews", "possible_event"])

    st.markdown("#### Political Events Associated With U.S. Pageview Peaks")
    st.dataframe(us_events_df, use_container_width=True)

    uk_events_df = pd.DataFrame([
        ["UK", "2023-05-06", "3.88M", "Coronation of King Charles III"],
        ["UK", "2023-11-13", "3.55M", "David Cameron returns as Foreign Secretary after cabinet reshuffle"],
        ["UK", "2024-07-05", "5.03M", "Day after 2024 general election; Labour landslide victory"],
        ["UK", "2024-11-06", "2.98M", "UK news dominated by global events (U.S. election, Middle East conflict)"]
    ], columns=["country", "date", "pageviews", "possible_event"])

    st.markdown("#### Political Events Associated With UK Pageview Peaks")
    st.dataframe(uk_events_df, use_container_width=True)

    
    india_events_df = pd.DataFrame([
        ["India", "2023-03-24", "4.07M", "Rahul Gandhi disqualified from Parliament after defamation conviction"],
        ["India", "2023-05-13", "5.58M", "Karnataka Assembly election results; Congress victory"],
        ["India", "2023-12-03", "5.76M", "Five-state election period; results announced Dec 4 with major BJP wins"],
        ["India", "2024-06-04", "11.48M", "2024 general election results; BJP loses majority, NDA forms coalition"],
        ["India", "2024-10-10", "5.42M", "Major political/diplomatic events including PM Modi’s Laos visit"]
    ], columns=["country", "date", "pageviews", "possible_event"])

    st.markdown("#### Political Events Associated With India Pageview Peaks")
    st.dataframe(india_events_df, use_container_width=True)

    australia_events_df = pd.DataFrame([
        ["Australia", "2023-05-06", "864k", "No major national event; routine political news and ongoing 2023 political developments"],
        ["Australia", "2023-10-14", "704k", "Indigenous Voice to Parliament referendum (national 'No' vote)"],
        ["Australia", "2024-11-06", "1.36M", "Federal Parliament activity; political controversies covered in national media; Possible coverage of US politics"]
    ], columns=["country", "date", "pageviews", "possible_event"])

    st.markdown("#### Political Events Associated With Australia Pageview Peaks")
    st.dataframe(australia_events_df, use_container_width=True)

    canada_events_df = pd.DataFrame([
        ["Canada", "2024-11-06", "2.34M", "Canadian government orders TikTok to cease operations; national discussion on U.S. election impact"]
    ], columns=["country", "date", "pageviews", "possible_event"])

    st.markdown("#### Political Events Associated With Canadian Pageview Peaks")
    st.dataframe(canada_events_df, use_container_width=True)