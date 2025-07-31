#!/usr/bin/env python
# coding: utf-8

# # **Analysis Project of the Largest Companies in the USA**
# 
# **Project Objective:**  
# To perform a comprehensive analysis on data of the largest 100 companies in the USA collected from Wikipedia.

# ## **1. Data Extraction via Web Scraping**
# 
# Company information was extracted from the Wikipedia page using web scraping with BeautifulSoup.

# In[1]:


from bs4 import BeautifulSoup  # Imports BeautifulSoup library to parse HTML and XML documents
import requests               # Imports requests library to make HTTP requests


# In[2]:


url = 'https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue'  
# Defines the URL of the web page from which data will be scraped

page = requests.get(url)  
# Sends an HTTP GET request to the URL and retrieves the page content

soup = BeautifulSoup(page.text, 'html')  
# Parses the HTML content of the page using BeautifulSoup  
# Using the 'html' parser to interpret the page structure


# In[3]:


print(soup)  
# Prints the parsed HTML content to the console to inspect the entire page structure


# In[4]:


soup.find('table')  
# Finds and returns the first <table> element in the HTML content  
# This table likely contains the list of the largest companies on the page


# In[5]:


soup.find_all('table')[0]  
# Finds all <table> elements on the page as a list  
# Selects the first table (index 0) from this list  
# This allows access to the first table on the page


# In[6]:


soup.find('table', class_='wikitable sortable')  
# Finds the table with the class 'wikitable sortable' in the HTML content  
# This is typically the class name for sortable tables on Wikipedia  
# We select the specific table this way for precision


# In[7]:


table = soup.find_all('table')[0]  
# Finds all tables on the page and assigns the first table to the variable 'table'  
# This allows us to start working with this specific table


# In[8]:


print(table)  
# Prints the selected table in HTML format to the console  
# Used to observe the table's content and structure


# ## **2. Processing Table Data and Saving as CSV**
# 
# Table headers and data were parsed and converted into a pandas DataFrame.  
# Then, the data was saved as a CSV file.

# In[9]:


soup.find_all('th')  
# Finds all <th> (table header) tags on the page and returns them as a list  
# These tags typically represent the column headers in the table


# In[10]:


world_titles = table.find_all('th')  
# Finds all header cells (<th> tags) within the selected table  
# These headers contain the column names and are assigned to the list 'world_titles'


# In[11]:


world_titles  
# Displays the table header cells stored in the 'world_titles' variable  
# Allows us to inspect the list of column headers


# In[12]:


world_table_titles = [title.text for title in world_titles]  
# Extracts the text from each header cell (<th>) and creates a new list  
# This results in a simple list of column header names

print(world_table_titles)  
# Prints the created list of headers to the console for verification


# In[13]:


world_table_titles = [title.text.strip() for title in world_titles]  
# Extracts the text from each header cell and removes leading/trailing whitespace  
# This creates a cleaner and more organized list of headers

print(world_table_titles)  
# Prints the cleaned header list to the console for verification


# In[14]:


import pandas as pd  
# Imports the pandas library for data analysis and table operations  
# pandas allows us to easily manipulate and analyze data using DataFrames


# In[15]:


df = pd.DataFrame(columns=world_table_titles)  
# Creates an empty DataFrame using the previously extracted headers as column names  
# We can add data to this table later

df  
# Displays the created empty DataFrame  
# It currently contains no data, only column headers


# In[16]:


table.find_all('tr')  
# Finds all rows (<tr> tags) within the table and returns them as a list  
# These rows include both header and data rows


# In[17]:


column_data = table.find_all('tr')  
# Finds all rows (<tr> tags) within the table and assigns them to the list 'column_data'  
# This list contains both header and data rows from the table


# In[18]:


for row in column_data:  
    # Processes each row in the table one by one

    row_data = row.find_all('td')  
    # Finds all data cells (<td> tags) within the row

    individual_row_data = [data.text.strip() for data in row_data]  
    # Extracts the text from each cell and strips leading/trailing whitespace

    print(individual_row_data)  
    # Prints the cleaned data from the row  
    # Header rows may be empty since they use <th> tags instead of <td>


# In[19]:


for row in column_data[1:]:  
    # Iterates over each data row in the table, excluding the first header row

    row_data = row.find_all('td')  
    # Finds all data cells (<td> tags) within the current row

    individual_row_data = [data.text.strip() for data in row_data]  
    # Extracts and cleans the text content of each cell

    print(individual_row_data)  
    # Prints the cleaned data for each row to verify the content


# In[20]:


for row in column_data[1:]:  
    # Iterates over each data row in the table, excluding the header row

    row_data = row.find_all('td')  
    # Finds all data cells within the current row

    individual_row_data = [data.text.strip() for data in row_data]  
    # Extracts and cleans the text content from each cell

    length = len(df)  
    # Gets the current number of rows in the DataFrame (used as the new row index)

    df.loc[length] = individual_row_data  
    # Adds the new row to the end of the DataFrame  
    # This way, table data is appended row by row into the DataFrame


# In[21]:


df  
# Displays the DataFrame containing the data  
# All rows extracted from the table in the previous loop are listed here


# In[22]:


df.to_csv(r'C:\Users\secki\OneDrive\Desktop\Projects\scrap_us_revenue\Companies.csv', index=False)  
# Saves the DataFrame as a CSV file to the specified file path  
# 'index=False' ensures that row indices are not included in the file  
# This allows exporting the data for use elsewhere


# ## **3. Loading Data and Initial Inspection**
# 
# The saved CSV file was loaded using pandas.  
# Data types and the first 5 rows were inspected.

# In[23]:


import pandas as pd

# Reads the CSV file
df = pd.read_csv(r'C:\Users\secki\OneDrive\Desktop\Projects\scrap_us_revenue\Companies.csv')

# Displays the first 5 rows
print(df.head())

# Provides general information about the data
print(df.info())

# Shows summary statistics (for numerical columns)
print(df.describe())


# ## **4. Data Cleaning and Numeric Conversion**
# 
# Thousand separators and percentage signs in columns that should be numeric were cleaned,  
# and data types were converted to appropriate numeric formats.

# In[24]:


# Removes thousand separators ',' from 'Revenue' and converts to float
df['Revenue (USD millions)'] = df['Revenue (USD millions)'].str.replace(',', '').astype(float)

# Removes '%' sign from 'Revenue growth' and converts to float
df['Revenue growth'] = df['Revenue growth'].str.replace('%', '').astype(float)

# Removes ',' from 'Employees' and converts to int
df['Employees'] = df['Employees'].str.replace(',', '').astype(int)

# Checks data types after conversion
print(df.dtypes)


# ## **5. Sector-Based Analysis and Visualization**
# 
# The number of companies, total revenue, and total employees were analyzed by sector and displayed with graphs.

# In[25]:


# Number of companies by sector
sector_counts = df['Industry'].value_counts()
print("Number of companies by sector:")
print(sector_counts)


# In[26]:


# Total revenue by sector
sector_revenue = df.groupby('Industry')['Revenue (USD millions)'].sum().sort_values(ascending=False)
print("\nTotal revenue by sector (million USD):")
print(sector_revenue)

import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style for better aesthetics in plots
sns.set(style="whitegrid")

# Plot total revenue by sector
plt.figure(figsize=(12, 8))
sector_revenue.plot(kind='bar')
plt.title('Total Revenue by Sector (Million USD)')
plt.ylabel('Total Revenue (Million USD)')
plt.xlabel('Sector')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[27]:


sector_employees = df.groupby('Industry')['Employees'].sum().sort_values(ascending=False)
print("\nTotal number of employees by sector:")
print(sector_employees)

plt.figure(figsize=(12, 8))
sector_employees.plot(kind='bar', color='orange')
plt.title('Total Number of Employees by Sector')
plt.ylabel('Total Number of Employees')
plt.xlabel('Sector')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ## **6. Revenue and Growth Rate Analysis**
# 
# Revenue distribution was examined with histogram and boxplot.  
# The mean, median, and positive/negative distribution of growth rates were shown.

# In[28]:


# Revenue distribution histogram
plt.figure(figsize=(10,6))
sns.histplot(df['Revenue (USD millions)'], bins=30, kde=True)
plt.title('Distribution of Company Revenues (Million USD)')
plt.xlabel('Revenue (Million USD)')
plt.ylabel('Number of Companies')
plt.show()

# Revenue distribution boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Revenue (USD millions)'])
plt.title('Boxplot of Company Revenues')
plt.show()

# Growth rate statistics
print("Average growth rate: {:.2f}%".format(df['Revenue growth'].mean()))
print("Median growth rate: {:.2f}%".format(df['Revenue growth'].median()))

# Count of positive and negative growth
pos_growth = (df['Revenue growth'] > 0).sum()
neg_growth = (df['Revenue growth'] < 0).sum()
print(f"Number of companies with positive growth: {pos_growth}")
print(f"Number of companies with negative growth: {neg_growth}")

# Top 5 fastest growing companies
print("\nTop 5 fastest growing companies:")
print(df[['Name', 'Revenue growth']].sort_values(by='Revenue growth', ascending=False).head())

# Top 5 fastest shrinking companies
print("\nTop 5 fastest shrinking companies:")
print(df[['Name', 'Revenue growth']].sort_values(by='Revenue growth').head())


# ## **7. Relationship Between Number of Employees and Revenue**
# 
# The relationship between number of employees and revenue was visualized with a scatter plot.  
# The correlation coefficient was calculated and displayed.

# In[29]:


# Scatter plot showing the relationship between number of employees and revenue
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Employees', y='Revenue (USD millions)')
plt.title('Relationship Between Number of Employees and Revenue')
plt.xlabel('Number of Employees')
plt.ylabel('Revenue (Million USD)')
plt.show()

# Correlation coefficient calculation
corr = df['Employees'].corr(df['Revenue (USD millions)'])
print(f"Correlation coefficient between number of employees and revenue: {corr:.2f}")


# ## **8. Geographic Distribution Analysis**
# 
# Headquarter state information was extracted.  
# The number of companies and total revenue by state were analyzed and visualized.

# In[30]:


# Extracts the state information from the 'Headquarters' column (assumed as the last word)
df['State'] = df['Headquarters'].apply(lambda x: x.split(',')[-1].strip())

# Number of companies by state
state_counts = df['State'].value_counts()

# Total revenue by state
state_revenue = df.groupby('State')['Revenue (USD millions)'].sum().sort_values(ascending=False)

# Plot: Number of companies by state
plt.figure(figsize=(12,6))
sns.barplot(x=state_counts.index, y=state_counts.values)
plt.title('Number of Companies by State')
plt.ylabel('Number of Companies')
plt.xlabel('State')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot: Total revenue by state
plt.figure(figsize=(12,6))
sns.barplot(x=state_revenue.index, y=state_revenue.values)
plt.title('Total Revenue by State (Million USD)')
plt.ylabel('Total Revenue (Million USD)')
plt.xlabel('State')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ## **9. Clustering Analysis**
# 
# Companies were clustered using K-Means based on revenue, growth, and number of employees.  
# Dimensionality reduction was performed using PCA and visualized in 2D.

# In[31]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Variables used for clustering analysis
X = df[['Revenue (USD millions)', 'Revenue growth', 'Employees']]

# Standardizes (scales) the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of clusters (example: 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Counts of each cluster
print(df['Cluster'].value_counts())

# Cluster-wise averages
print(df.groupby('Cluster')[['Revenue (USD millions)', 'Revenue growth', 'Employees']].mean())


# In[32]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Cluster'], palette='Set2', s=100)
plt.title('Company Distribution by Clusters (2D Visualization with PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()


# ## **10. Sector Distribution by Cluster**
# 
# The distribution of sectors within each cluster was analyzed using a heatmap.

# In[33]:


cluster_sector_counts = df.groupby(['Cluster', 'Industry']).size().unstack(fill_value=0)

# Number of companies by sector within each cluster
print(cluster_sector_counts)

# Visualization as a heatmap
plt.figure(figsize=(14,8))
sns.heatmap(cluster_sector_counts, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Number of Companies by Clusters and Sectors (Heatmap)')
plt.ylabel('Cluster')
plt.xlabel('Industry')
plt.show()


# ## **11. Performance Analysis of the Top 10 Largest Companies**
# 
# Revenue, growth rate, and number of employees were visualized using graphs.

# In[34]:


# Top 10 largest companies by revenue
top10 = df.sort_values(by='Revenue (USD millions)', ascending=False).head(10)

print(top10[['Name', 'Revenue (USD millions)', 'Revenue growth', 'Employees']])

# Plot: Revenue and growth rate of the top 10 companies
fig, ax1 = plt.subplots(figsize=(12,6))

color = 'tab:blue'
ax1.set_xlabel('Company')
ax1.set_ylabel('Revenue (Million USD)', color=color)
ax1.bar(top10['Name'], top10['Revenue (USD millions)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=45, ha='right')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Growth Rate (%)', color=color)
ax2.plot(top10['Name'], top10['Revenue growth'], color=color, marker='o', linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Revenue and Growth Rate of the Top 10 Companies')
plt.tight_layout()
plt.show()

# Plot: Number of employees in the top 10 companies
plt.figure(figsize=(12,6))
sns.barplot(x=top10['Name'], y=top10['Employees'])
plt.title('Number of Employees in the Top 10 Companies')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Employees')
plt.show()


# ## **12. Conclusions and Recommendations**
# 
# This analysis provided an in-depth examination of data related to the largest companies in the USA.  
# Key insights and recommendations are as follows:
# 
# - **Sector Concentration:** The petroleum, healthcare, and technology sectors stand out in terms of both the number of companies and total revenue.  
# - **Revenue and Employee Count:** There is a strong positive correlation of approximately **0.7** between revenue and number of employees, indicating that larger companies tend to employ more people.  
# - **Growth Trends:** A significant portion of companies show positive growth, although some sectors exhibit shrinking trends.  
# - **Geographic Distribution:** Texas, New York, and California host the highest number of large companies.  
# - **Clustering Results:** Companies were grouped into **4 meaningful clusters** based on revenue, growth, and employee count:  
#   - **Clusters 0 and 1:** Medium-sized companies with positive growth, employing roughly 150-160 thousand people.  
#   - **Cluster 2:** Very large companies with
