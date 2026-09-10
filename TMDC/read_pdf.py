import PyPDF2

# 打开PDF文件
with open('Paper-TMDC.pdf', 'rb') as file:
    # 创建PDF阅读器对象
    reader = PyPDF2.PdfReader(file)
    
    # 获取PDF的页数
    num_pages = len(reader.pages)
    print(f"PDF总页数: {num_pages}")
    
    # 将内容保存到文本文件
    with open('paper_content.txt', 'w', encoding='utf-8') as output_file:
        # 逐页读取内容
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            output_file.write(f"\n=== 第 {page_num + 1} 页 ===\n")
            output_file.write(text)
    
    print("PDF内容已保存到paper_content.txt文件中")
