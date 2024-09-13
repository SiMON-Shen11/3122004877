import sys
import paper_check

# 获取命令行参数的辅助函数
def fetch_parameters():
    parameters = []
    try:
        if len(sys.argv) < 4:
            raise ValueError("参数数量不足，请输入3个参数！")
        for i in range(1, 4):
            if isinstance(sys.argv[i], str):
                parameters.append(sys.argv[i])
            else:
                raise TypeError("参数类型错误，所有参数应为字符串！")
    except ValueError as ve:
        print(ve)
        return None
    except TypeError as te:
        print(te)
        return None
    return parameters


# 主体算法逻辑
def check_paper_similarity():
    # 文件路径配置
    original_file = "./test_text/orig.txt"
    compared_file1 = "./test_text/orig.txt"
    compared_file2 = "./test_text/orig_0.8_add.txt"
    compared_file3 = "./test_text/orig_0.8_del.txt"
    compared_file4 = "./test_text/orig_0.8_dis_1.txt"
    compared_file5 = "./test_text/orig_0.8_dis_10.txt"
    compared_file6 = "./test_text/orig_0.8_dis_15.txt"
    output_result = "./test_text/result.txt"

    # 可以通过命令行参数传入文件路径（若需要）
    # params = fetch_parameters()
    # original_file, comparison_file, output_file = params[0], params[1], params[2]

    # 处理文件内容并去除标点符号
    doc1, doc2 = paper_check.load_and_process_files(original_file, compared_file2)
    # 对文件内容进行分词
    tokens1, tokens2 = paper_check.tokenize_content(doc1, doc2)
    # 分词内容转化为向量，进行降维分析
    feature_vector1, feature_vector2 = paper_check.convert_to_vector(tokens1, tokens2)
    # 计算余弦相似度
    cosine_similarity = paper_check.compute_cosine_similarity(feature_vector1, feature_vector2)
    # 输出相似度结果
    print(f"文档相似度为: {cosine_similarity}")
    paper_check.save_similarity_result(output_result, cosine_similarity)


# 程序的入口
if __name__ == '__main__':
    check_paper_similarity()
