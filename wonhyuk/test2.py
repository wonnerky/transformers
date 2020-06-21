import wget, pickle, time
import data_processing
from transformers import BartTokenizer

file_path = '.\\data\\preprocessing\\wiki103\\'
tokenizer = BartTokenizer.from_pretrained('bart-large')
# start_time = time.time()
# wiki = data_processing.WikiDataset(tokenizer=tokenizer, file_path=file_path, block_size=16, data_type='train', flag=1)
# print(time.time() - start_time)
# print(wiki)
# print(wiki[0])

with open(f'{file_path}test_one_sentence_keyword.txt', 'rb') as f:
    keyword_list = pickle.load(f)
print(keyword_list[0])



for i in range(10):
    wiki_keyword = data_processing.WikiKeywordDataset(tokenizer=tokenizer, file_path=file_path, block_size=128, data_type='test', flag=1, shuffle=True)
    print(wiki_keyword[0])
    print(tokenizer.decode(wiki_keyword[0]))
