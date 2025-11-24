import os

file_path = "../data"

if os.path.exists(file_path):
    print(f"Đường dẫn '{file_path}' tồn tại.")
    if os.path.isfile(file_path):
        print(f"'{file_path}' là file.")
    elif os.path.isdir(file_path):
        print(f"'{file_path}' là thư mục.")
else:
    print(f"Đường dẫn '{file_path}' không tồn tại.")
