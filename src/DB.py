# -*- coding: utf-8 -*-

from __future__ import print_function

import pandas as pd
import os

DB_dir = 'database' #đường dẫn đến thư mục chứa dữ liệu
DB_csv = 'data.csv' # đường dẫn đến file .csv chứa metadata của dữ liệu


class Database(object):

  def __init__(self):
    self._gen_csv() # để tạo file .csv nếu chưa tồn tại.
    self.data = pd.read_csv(DB_csv) #Đọc file .csv
    self.classes = set(self.data["cls"]) # Lấy tập hợp giá trị cột "cls" và lưu vào thuộc tính classes.

  def _gen_csv(self):
    if os.path.exists(DB_csv): # file.csv đã tồn tại
      return
    # duyệt database, lấy tên file .jpg và ghi vào file .csv dạng "img,cls".
    with open(DB_csv, 'w', encoding='UTF-8') as f:
      f.write("img,cls")
      for root, _, files in os.walk(DB_dir, topdown=False):
        cls = root.split('/')[-1]
        for name in files:
          if not name.endswith('.jpg'):
            continue
          img = os.path.join(root, name)
          f.write("\n{},{}".format(img, cls))

#Trả về độ dài dataframe data
  def __len__(self):
    return len(self.data)

#Trả về tập hợp giá trị cột "cls".
  def get_class(self):
    return self.classes

  def get_data(self):
    return self.data


if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  classes = db.get_class()

  print("DB length:", len(db))
  print(classes)
