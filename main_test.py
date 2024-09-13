import unittest
import numpy as np
import paper_check
import main

class PaperCheckTest(unittest.TestCase):

    # 测试读取空文件
    def test_read_empty_file(self):
        orig_path = ""  # 空文件路径
        orig_add_path = "./test_text/orig_0.8_add.txt"
        result = paper_check.load_and_process_files(orig_path, orig_add_path)
        self.assertEqual(result, (None, None), "读取空文件时应该返回 (None, None)")

    # 测试读取不存在的文件
    def test_read_not_exist_file(self):
        orig_path = "./test_text/non_existent_file.txt"
        orig_add_path = "./test_text/orig_0.8_add.txt"
        result = paper_check.load_and_process_files(orig_path, orig_add_path)
        self.assertEqual(result, (None, None), "读取不存在的文件时应该返回 (None, None)")

    # 测试读取的文件能否过滤标点符号
    def test_filter_dot(self):
        orig_path = "./test_text/orig.txt"
        orig_add_path = "./test_text/orig_0.8_add.txt"
        orig_file, orig_add_file = paper_check.load_and_process_files(orig_path, orig_add_path)
        self.assertNotIn("。", orig_file, "标点符号未正确过滤")
        self.assertNotIn("。", orig_add_file, "标点符号未正确过滤")
        print(orig_file, orig_add_file)

    # 测试文件内容分词效果
    def test_tokenize_content(self):
        orig_path = "./test_text/orig.txt"
        orig_add_path = "./test_text/orig_0.8_add.txt"
        orig_file, orig_add_file = paper_check.load_and_process_files(orig_path, orig_add_path)
        orig_string, orig_add_string = paper_check.tokenize_content(orig_file, orig_add_file)
        self.assertIsInstance(orig_string, str, "分词结果应为字符串")
        self.assertIsInstance(orig_add_string, str, "分词结果应为字符串")
        print(orig_string, orig_add_string)

    # 测试分词向量化效果
    def test_convert_to_vector(self):
        orig_string = '今天 星期天 天气 晴 舒适 看电影'
        orig_add_string = '今日 周天 天气 晴朗 愉快 看小说'
        vector1, vector2 = paper_check.convert_to_vector(orig_string, orig_add_string)
        self.assertEqual(len(vector1), len(vector2), "向量化后两个向量的长度应该相同")
        print(vector1, vector2)

    # 测试余弦相似度计算
    def test_calculate_cosine_similarity(self):
        vector1 = np.array([1, 2, 3, 4, 5, 6])
        vector2 = np.array([1, 2, 3, 4, 5, 6])
        similarity = paper_check.compute_cosine_similarity(vector1, vector2)
        self.assertEqual(similarity, 1.0, "完全相同的向量余弦相似度应为 1")
        print(similarity)

    # 测试输出结果到指定文件中
    def test_save_similarity_result(self):
        answer_path = "./test_text/result02.txt"
        result = 0.86748569623
        saved_result = paper_check.save_similarity_result(answer_path, result)
        self.assertEqual(saved_result, round(result, 2), "保存的相似度结果应保留两位小数")
        print(saved_result)

    # 测试输出结果到不存在文件中
    def test_save_not_exist_file(self):
        answer_path = ""  # 空路径
        result = 0.86748569623
        saved_result = paper_check.save_similarity_result(answer_path, result)
        self.assertIsNone(saved_result, "保存到不存在的文件路径时应返回 None")

    # 测试输出的结果不是浮点型
    def test_result_not_float(self):
        answer_path = "./test_text/result03.txt"
        result = 2  # 非浮点型
        try:
            saved_result = paper_check.save_similarity_result(answer_path, result)
        except ValueError as e:
            print(f"捕获到错误: {e}")  # 打印错误信息，提示发生了什么问题
            saved_result = None  # 将返回值设置为 None 以继续测试流程
    
        self.assertIsNone(saved_result, "当结果不是浮点型时，应返回 None 或抛出 ValueError。")


    # 测试文章查重的整体流程
    def test_paper_checked(self):
        main.check_paper_similarity()
        # 无具体返回值，通常检查文件是否生成或结果是否正确


if __name__ == '__main__':
    unittest.main()
