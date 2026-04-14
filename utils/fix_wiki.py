# fix_wiki.py
import os

file_path = "../wikiextractor/wikiextractor/extract.py"

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定义需要修复的错误特征字符串（Python 3.11+ 不允许 global flags 在中间）
    bad_str = '[^\\s\\x00-\\x1f\\\\<>"\x27]+(?i)'
    good_str = '(?i)[^\\s\\x00-\\x1f\\\\<>"\x27]+'

    if bad_str in content:
        new_content = content.replace(bad_str, good_str)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ WikiExtractor 正则表达式修复完成 (String Replace 模式)")
    else:
        print("ℹ️ 未发现匹配项，可能已经修复过或源码版本不同")
else:
    print(f"❌ 未找到文件: {file_path}")