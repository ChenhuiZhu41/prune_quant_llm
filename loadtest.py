from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

#模型本地路径
local_model_path = "/home/zhuchenhui/wanda/Llama3-1b"
print("Model and tokenizer loaded start")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,  # 自动选择数据类型（如 float16）
    low_cpu_mem_usage=True,  # 减少 CPU 内存使用
    device_map="cuda:0"
)

# 测试加载成功
print("Model and tokenizer loaded successfully!")
print("try to load wikitext datasets")
# en = load_dataset("allenai/c4", "en")
tokenizer = AutoTokenizer.from_pretrained("/home/zhuchenhui/wanda/Llama3-1b")
traindata = load_dataset(path='parquet', data_files='/home/zhuchenhui/dataset/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet', split='train')
testdata = load_dataset(path='parquet', data_files='/home/zhuchenhui/dataset/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet')
print(testdata)
# trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
testenc = tokenizer("\n\n".join(testdata['train']['text']), return_tensors='pt')
print("load c4 datasets successfully")