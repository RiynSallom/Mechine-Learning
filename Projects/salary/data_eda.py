# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 00:09:55 2022

@author: Rayan
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv('salary_data_cleaned.csv')
df.head()
df.columns
def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'
df['job_simp'] = df['Job Title'].apply(title_simplifier)
df.job_simp.value_counts()

df['seniority'] = df['Job Title'].apply(seniority)
df.seniority.value_counts()

df['job_state']= df.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
df.job_state.value_counts()

df['desc_len'] = df['Job Description'].apply(lambda x: len(x))
df['desc_len']

df['num_comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)

df['Competitors']

df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly ==1 else x.min_salary, axis =1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly ==1 else x.max_salary, axis =1)

df[df.hourly ==1][['hourly','min_salary','max_salary']]
df['company_txt']
df['company_txt'] = df.company_txt.apply(lambda x: x.replace('\n', ''))

df['company_txt']

df.describe()

df.columns

df.Rating.hist()

df.avg_salary.hist()

df.age.hist()

df.desc_len.hist()

df.boxplot(column = ['age','avg_salary','Rating'])

df.boxplot(column = 'Rating')

df[['age','avg_salary','Rating','desc_len']].corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df[['age','avg_salary','Rating','desc_len','num_comp']].corr(),vmax=.3, center=0, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

df.columns

df_cat = df[['Location', 'Headquarters', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 'company_txt', 'job_state','same_state', 'python_yn', 'R_yn',
       'spark', 'aws', 'excel', 'job_simp', 'seniority']]

for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()
    
for i in df_cat[['Location','Headquarters','company_txt']].columns:
    cat_num = df_cat[i].value_counts()[:20]
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()
    
    
pd.pivot_table(df, index = 'job_simp', values = 'avg_salary')

pd.pivot_table(df, index = ['job_simp','seniority'], values = 'avg_salary')

pd.pivot_table(df, index = ['job_state','job_simp'], values = 'avg_salary').sort_values('job_state', ascending = False)

pd.options.display.max_rows
pd.set_option('display.max_rows', None)

pd.pivot_table(df, index = ['job_state','job_simp'], values = 'avg_salary', aggfunc = 'count').sort_values('job_state', ascending = False)

pd.pivot_table(df[df.job_simp == 'data scientist'], index = 'job_state', values = 'avg_salary').sort_values('avg_salary', ascending = False)

df.columns

df_pivots = df[['Rating', 'Industry', 'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided', 'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'Type of ownership','avg_salary']]

for i in df_pivots.columns:
    print(i)
    if(i  != 'avg_salary'):
        print(pd.pivot_table(df_pivots,index =i, values = 'avg_salary').sort_values('avg_salary', ascending = False))

pd.pivot_table(df_pivots, index = 'Revenue', columns = 'python_yn', values = 'avg_salary', aggfunc = 'count')

df.to_csv("eda_data.csv",index=False)