from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

url = "https://www.imdb.com/chart/top/"
driver.get(url)
time.sleep(3)  

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

movies = []

rows = soup.select('li.ipc-metadata-list-summary-item')  
print(f"총 항목 수: {len(rows)}")  

for row in rows:
    title_tag = row.select_one('a.ipc-title-link-wrapper')  # 제목이 포함된 <a> 태그
    title = title_tag.text.strip() if title_tag else "Unknown Title"
    if title and "." in title.split(" ")[0]:
        title = " ".join(title.split(" ")[1:])  # 제목에서 순위 제거
    movie_url = "https://www.imdb.com" + title_tag['href'] if title_tag else None

    meta_data = row.select('span.sc-300a8231-7')  
    year = meta_data[0].text.strip() if len(meta_data) > 0 else "Unknown Year"  
    runtime = meta_data[1].text.strip() if len(meta_data) > 1 else "Unknown Runtime" 

    if movie_url:
        driver.get(movie_url)
        time.sleep(2) 
        movie_html = driver.page_source
        movie_soup = BeautifulSoup(movie_html, 'html.parser')

        description_tag = movie_soup.select_one('span.sc-3ac15c8d-0')  # 영화 설명
        description = description_tag.text.strip() if description_tag else "Unknown Description"

        director_tag = movie_soup.select_one('li.ipc-metadata-list__item a.ipc-metadata-list-item__list-content-item')
        director = director_tag.text.strip() if director_tag else "Unknown Director"

        cast_tags = movie_soup.select('li.ipc-inline-list__item a.ipc-metadata-list-item__list-content-item')[:5]  # 주요 배우
        cast = [tag.text.strip() for tag in cast_tags]

        movies.append({
            "title": title,
            "year": year,
            "runtime": runtime,
            "director": director,
            "cast": ", ".join(cast),
            "description": description,
            "url": movie_url
        })

driver.quit()

movies_df = pd.DataFrame(movies)
movies_df.to_csv('./crawling_data/movies_metadata.csv', index=False)

print("크롤링 완료 및 movies_metadata.csv 생성 완료!")
