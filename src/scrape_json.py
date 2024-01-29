import requests
from bs4 import BeautifulSoup
import os
import json

content = []
links = []
dones = set()
heads = set()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 '
                  'Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br'}

root_url = "https://docs.llamaindex.ai/en/stable/"
with open(os.path.join('base','LlamaIndex.html'), 'r', encoding='utf-8') as f:
    pri_file = f.read()
error_urls = set()


def process_page(url: str, page: str) -> tuple:
    page_content = []
    global links, dones, heads
    parser = BeautifulSoup(page, 'html.parser')
    links_sections = parser.find_all('a', href=True)
    urls = [link['href'] for link in links_sections if
            link['href'] != "#" and not link['href'].startswith('https') and not link['href'].startswith('#') and link
            != root_url and not link['href'].startswith('..') and link not in ['/', '//', '//readthedocs.com',
                                                                        '///readthedocs.com', '////readthedocs.com']]
    text_sections = parser.find_all('div', class_='section')
    h1 = None
    for t in text_sections:
        h = t.find_all('h1')
        if h:
            h1 = h[0].text
            heads.add(h1)
        page_content.append(t.text)
    dones.add(url)
    links += urls
    return h1, page_content


process_page(root_url, pri_file)
print('Remaining links to process:', set(links) - dones)
while len(set(links) - dones) > 0:
    for link in (set(links) - dones):
        print('Link being processed:', link)
        url = root_url + link
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response = response.text
            header, page_contents = process_page(url, response)
            if not header:
                print('Error in header while processing link')
            content_string = '\n'.join(page_contents)
            header = header.encode('ascii', 'ignore').decode('ascii')
            header = ''.join([c for c in header if c not in ['//', '\/', '%', '?', '+', '/']])
            details = {'header':header, 'text':content_string, 'url':url}
            filename = header + '.json'
            with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
                json.dump(details,f)
        else:
            if response.status_code != 404:
                print(f'Failure: {url}\t Status code: {response.status_code}')
        dones.add(link)

    print('Remaining links to process:', set(links) - dones)
    print('\n')
pending = "None :)" if len(set(links) - dones) == 0 else set(links) - dones
print('Remaining links to process:', pending)
error_urls = 'None :)' if len(error_urls)>0 else error_urls
print('Errored out urls:', error_urls)
if error_urls:
    with open('error_urls.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(error_urls))
