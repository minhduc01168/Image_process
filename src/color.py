# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import distance, evaluate_class
from DB import Database
# import argparse module for argument parsing
import argparse
from six.moves import cPickle
import numpy as np
import scipy.misc
import itertools
import os
import cv2


# # configs for histogram
n_bin   = 12        # histogram bins
n_slice = 3         # slice image
h_type  = 'region'  # global or region
d_type  = 'd1'      # distance type


depth   = 3          # retrieved depth, set to None will count the ap for whole database

# # initialize the argument parser
# ap = argparse.ArgumentParser()
#
# # add argument for query image path
# ap.add_argument("-b", "--n_bin", required=True, help="histogram bins")
# ap.add_argument("-q", "--n_slice", required=True, help="slice image")
# ap.add_argument("-q", "--query_idx", required=True, help="index query")
# ap.add_argument("-q", "--query_idx", required=True, help="index query")
# # parse the arguments
# args = vars(ap.parse_args())
''' MMAP
     depth
      depthNone, region,bin12,slice3, distance=d1, MMAP 0.35371454855515705
      depth100,  region,bin12,slice3, distance=d1, MMAP 0.47422743076365287
      depth30,   region,bin12,slice3, distance=d1, MMAP 0.6025190664890758
      depth10,   region,bin12,slice3, distance=d1, MMAP 0.7043049915820964
      depth5,    region,bin12,slice3, distance=d1, MMAP 0.7354904053671018
      depth3,    region,bin12,slice3, distance=d1, MMAP 0.750727393335166
      depth1,    region,bin12,slice3, distance=d1, MMAP 0.7078575434858376

     (exps below use depth=None)
     
     d_type 
      global,bin6,d1,MMAP 0.23927699868270388
      global,bin6,cosine,MMAP 0.18859663332172952

     n_bin
      region,bin10,slice4,d1,MMAP 0.3592546290239438
      region,bin12,slice4,d1,MMAP 0.3596596412792158

      region,bin6,slcie3,d1,MMAP 0.339079596083966
      region,bin12,slice3,d1,MMAP 0.35371454855515716

     n_slice
      region,bin12,slice2,d1,MMAP 0.34773877187724533
      region,bin12,slice3,d1,MMAP 0.353714548555157
      region,bin12,slice4,d1,MMAP 0.3596596412792158
      region,bin14,slice3,d1,MMAP 0.3470500152263855
      region,bin14,slice5,d1,MMAP 0.3593876449758829
      region,bin16,slice3,d1,MMAP 0.35160582070492413
      region,bin16,slice4,d1,MMAP 0.3575296187223664
      region,bin16,slice8,d1,MMAP 0.34305960171438726

     h_type
      region,bin4,slice2,d1,MMAP 0.2892155340992675
      global,bin4,d1,MMAP 0.21397349694791434
'''

# t???o m???t th?? m???c m???i c?? t??n l?? "cache" n???u n?? ch??a t???n t???i.
# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Color(object):

  def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img color histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray( ???nh ?????u v??o)
        n_bin    : number of bins for each channel( s??? l?????ng bins)
        type     : 'global' means count the histogram for whole image( lo???i t??nh histogram: to??n b???, t???ng v??ng)
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices (s??? l?????ng v??ng ???????c chia)
        normalize: normalize output histogram ( N???u "normalize" l?? True, k???t qu??? histogram s??? ???????c chu???n h??a.)
  
      return
        type == 'global'
          a numpy array with size n_bin ** channel
        type == 'region'
          a numpy array with size n_slice * n_slice * (n_bin ** channel)
    '''
    if isinstance(input, np.ndarray):  # examinate input type ( n???u ?????u v??o l?? numpy)
      img = input.copy()
    else:# ?????u v??o l?? ???nh
      img = cv2.imread(input) # ?????c ???nh
    height, width, channel = img.shape # l???y chi???u d??i, chi???u r???ng, s??? k??nh
    #t???o m???ng bins b???ng c??ch chia nh??? ??o???n gi?? tr??? t??? 0 ?????n 256 th??nh n_bin+1 ph???n b???ng nhau
    bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel
    # t??nh histogram tren toan bo anh
    if type == 'global':
      hist = self._count_hist(img, n_bin, bins, channel)
    # t??nh histogram tren tung phan anh
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin ** channel))# t???o m???ng 3 chi???u l???n l?????t l?? n_slice, n_slice, n_bin**channel ????? l??u c??c v??ng
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)# c??c ??i???m chia ?????u chi???u cao
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)# c??c ??i???m chia ?????u chi???u r???ng

      # duy???t qua t???ng v??ng
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions( c???t ???nh th??nh c??c v??ng)
          hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)# t??nh histogram t???ng v??ng

    # n???u normalize=True, hist/ t???ng ????? chu???n h??a histogram.
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  # t??nh histogram
  def _count_hist(self, input, n_bin, bins, channel):
    img = input.copy()
    # dictionary bins_idx l??u tr??? ch??? m???c c???a c??c bin
    bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
    hist = np.zeros(n_bin ** channel)# l??u tr??? s??? l?????ng pixel cho m???i bin

    # ????a pixel v??o t???ng bins
    # cluster every pixels
    for idx in range(len(bins)-1):
      img[(input >= bins[idx]) & (input < bins[idx+1])] = idx # g??n t???ng gi?? tr??? m??u thu???c 1 kho???ng th??nh gtri bin c???a kho???ng ????
    # add pixels into bins
    height, width, _ = img.shape
    for h in range(height):
      for w in range(width):
        b_idx = bins_idx[tuple(img[h,w])]# t??? gi?? tr??? m??u ---> ch??? m???c c???a hist( gtri bin)
        hist[b_idx] += 1 # v???i gi?? tr??? bin ???? c???ng d???n v??o s??? l?????ng c???a bin ????
  
    return hist
  
  # h??m t???o ra t??n cache d???a v??o tham s??? truy???n v??o (h_type, n_bin, n_slice).
  # Sau ????, check cache c?? t???n t???i hay kh??ng.
  # N???u cache ???? t???n t???i, n?? s??? s??? d???ng cache ???? v?? in ra th??ng b??o.
  # N???u cache kh??ng t???n t???i, n?? s??? t???o ra m???u t??? database b???ng h??m histogram() tr??n m???i ph???n t??? trong db v?? l??u k???t qu??? v??o cache.
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "histogram_cache-{}-n_bin{}".format(h_type, n_bin)
    elif h_type == 'region':
      sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}".format(h_type, n_bin, n_slice)
    
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
      samples = []# samples g???m 3 th??nh ph???n: 'img' (???nh), 'cls' (l???p c???a ???nh), 'hist' (histogram t????ng ???ng c???a v??ng)
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples


if __name__ == "__main__":
  db = Database()
  # data = db.get_data()# l???y d??? li???u
  # color = Color()

  # # test normalize
  # hist = color.histogram(data.ix[0,0], type='global')# t??nh histogram
  # assert hist.sum() - 1 < 1e-9, "normalize false" # Ki???m tra xem t???ng histogram chu???n h??a ch??a. N???u ch??a, in ra "normalize false".
  #
  # # test histogram bins
  # def sigmoid(z): # h??m sigmoid
  #   a = 1.0 / (1.0 + np.exp(-1. * z))
  #   return a
  # np.random.seed(0)
  # IMG = sigmoid(np.random.randn(2,2,3)) * 255
  # IMG = IMG.astype(int)
  # hist = color.histogram(IMG, type='global', n_bin=4)
  # assert np.equal(np.where(hist > 0)[0], np.array([37, 43, 58, 61])).all(), "global histogram implement failed"
  # hist = color.histogram(IMG, type='region', n_bin=4, n_slice=2)
  # assert np.equal(np.where(hist > 0)[0], np.array([58, 125, 165, 235])).all(), "region histogram implement failed"
  #
  # # examinate distance
  # np.random.seed(1)
  # IMG = sigmoid(np.random.randn(4,4,3)) * 255
  # IMG = IMG.astype(int)
  # hist = color.histogram(IMG, type='region', n_bin=4, n_slice=2)
  # IMG2 = sigmoid(np.random.randn(4,4,3)) * 255
  # IMG2 = IMG2.astype(int)
  # hist2 = color.histogram(IMG2, type='region', n_bin=4, n_slice=2)
  # assert distance(hist, hist2, d_type='d1') == 2, "d1 implement failed"
  # assert distance(hist, hist2, d_type='d2-norm') == 2, "d2 implement failed"

  # evaluate database( ????nh gi?? c?? s??? d??? li???u)
  APs = evaluate_class(db, f_class=Color, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
