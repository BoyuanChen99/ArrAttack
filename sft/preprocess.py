'''
首先构造instruction dataset，json格式输出
数据格式为instruction：
         input：
         response：
'''

import csv
import json

# csv_filename = '/path/to/...'

# data = []
# with open(csv_filename, 'r', encoding='utf-8') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         input_data = row[0]  
#         response_data = row[1] 
#         data.append({'input': input_data, 'response': response_data})

# instruction = "Please rephrase the following prompt to make it more discreet and robust, retaining its original meaning, and directly return the rephrased prompt."

# json_data = []
# for item in data:
#     json_data.append({'instruction': instruction, 'input': item['input'], 'response': item['response']})

# json_filename = '/path/to/train/instruction_dataset.json'
# with open(json_filename, 'w', encoding='utf-8') as file:
#     json.dump(json_data, file, ensure_ascii=False, indent=4)

import json
from sklearn.model_selection import train_test_split

with open("/path/to/data/train/instruction_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

with open("/path/to/data/train/train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("/path/to/data/train/val_data.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)



