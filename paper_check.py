import math
import logging
import jieba as tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 禁用 jieba 的日志
tokenizer.setLogLevel(logging.ERROR)

# 读取并处理文件内容
def load_and_process_files(filepath1, filepath2):
    try:
        # 读取文件内容
        with open(filepath1, 'r', encoding='utf-8') as file1, open(filepath2, 'r', encoding='utf-8') as file2:
            content1 = file1.read()
            content2 = file2.read()
    except FileNotFoundError:
        print("文件路径错误，请检查输入的文件路径！")
        return None, None
    
    # 去除文件中的标点符号
    punctuation = "\n\r 、，。；：？‘’“”''""！《》,.;:?!<>"
    for p in punctuation:
        content1 = content1.replace(p, '')
        content2 = content2.replace(p, '')
    
    return content1, content2


# 分词函数
def tokenize_content(content1, content2):
    # 使用 jieba 分词，将内容进行分词处理
    tokens1 = ' '.join(tokenizer.lcut(content1))
    tokens2 = ' '.join(tokenizer.lcut(content2))
    return tokens1, tokens2


# 文本向量化
def convert_to_vector(text1, text2):
    # 利用 TF-IDF 向量化分词后的文本内容
    vectorizer = TfidfVectorizer()
    # 输入两段文本进行向量化
    transformed_vectors = vectorizer.fit_transform([text1, text2]).toarray()
    return transformed_vectors[0], transformed_vectors[1]


# 计算向量的余弦相似度
def compute_cosine_similarity(vec1, vec2):
    # 计算向量的点积和模的乘积
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
    # 计算并返回余弦相似度
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


# 将结果保存到文件中
def save_similarity_result(output_path, similarity_score):
    try:
        if isinstance(similarity_score, float):
            with open(output_path, 'w') as file:
                # 保留两位小数写入文件
                file.write(f'{similarity_score:.2f}')
            return round(similarity_score, 2)
        else:
            raise ValueError("结果应为浮点类型！")
    except FileNotFoundError:
        print("结果保存路径不存在，请检查路径！")
        return None
