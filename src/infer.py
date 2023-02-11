# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

from color import Color
from gabor import Gabor
from HOG import HOG

depth = 5
d_type = 'd1'
query_idx = 0

if __name__ == '__main__':
  db = Database()

  # retrieve by color
  method = Color()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)


  # # retrieve by gabor
  # method = Gabor()
  # samples = method.make_samples(db)
  # query = samples[query_idx]
  # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # print(result)
  #
  # # retrieve by HOG
  # method = HOG()
  # samples = method.make_samples(db)
  # query = samples[query_idx]
  # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # print(result)

