#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import os
import re
import urllib.error
from os.path import basename
from urllib.parse import urlsplit
import requests


url = 'https://www.zhihu.com/question/40811570'

if not os.path.exists('images'):
    os.mkdir("images")

page_size = 50
offset = 0
url_content = urllib.request.urlopen(url).read().decode('utf-8')
answers = re.findall(r'h3 data-num="(.*?)"', url_content)
# limits = int(answers[0])
limits = 50

while offset < limits:
    post_url = "http://www.zhihu.com/node/QuestionAnswerListV2"
    params = json.dumps({
        'url_token': 40811570,
        'pagesize': page_size,
        'offset': offset
    })
    data = {
        '_xsrf': '',
        'method': 'next',
        'params': params
    }
    header = {
        'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0",
        'Host': "www.zhihu.com",
        'Referer': url
    }
    response = requests.post(post_url, data=data, headers=header)
    answer_list = response.json()["msg"]
    img_urls = re.findall('img .*?src="(.*?_b.*?)"', ''.join(answer_list))
    for img_url in img_urls:
        try:
            img_data = urllib.request.urlopen(img_url).read()
            file_name = basename(urlsplit(img_url)[2])
            output = open('images/' + file_name, 'wb')
            output.write(img_data)
            output.close()
        except:
            pass
    offset += page_size