import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

def scrape_jobs():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'}

    webpage = requests.get("https://www.ambitionbox.com/list-of-companies?page=1", headers=headers).text
    soup = BeautifulSoup(webpage, 'lxml')

    company = soup.find_all('div', class_='companyCardWrapper')

    names = []
    ratings = []
    reviews = []
    company_type = []
    employees = []
    company_category = []
    est = []
    hq = []

    for i in company:
        name = i.find('h2', class_="companyCardWrapper__companyName").text.strip()
        names.append(name)
        ratings.append(i.find('span', class_="companyCardWrapper__companyRatingValue").text.strip())

        reviews_title = i.find('span', class_='companyCardWrapper__ActionTitle', string='Reviews')
        for r in reviews_title:
            parent_a_tag = r.find_parent('a', class_='companyCardWrapper__ActionWrapper')
            review = parent_a_tag.find('span', class_='companyCardWrapper__ActionCount')
            reviews.append(review.text.strip())

        company_info = i.find('span', class_='companyCardWrapper__interLinking').text.strip().split('|')

        try:
            company_type.append(company_info[0])
        except:
            company_type.append(np.nan)

        try:
            employees.append(company_info[1])
        except:
            employees.append(np.nan)

        try:
            company_category.append(company_info[2])
        except:
            company_category.append(np.nan)

        try:
            est.append(company_info[3])
        except:
            est.append(np.nan)

        try:
            hq.append(company_info[4])
        except:
            hq.append(np.nan)

    # Creating a dataframe from the given values
    d = {'name': names, 'ratings': ratings, 'review': reviews, 'type': company_type, 'employees': employees,
         'category': company_category, 'Established': est, 'Headquatered': hq}
    df = pd.DataFrame(d)

    print(df)
    return df
