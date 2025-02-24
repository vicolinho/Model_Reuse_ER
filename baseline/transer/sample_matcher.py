import os

from baseline.transer import model

src_dataset ='DBLP-ACM'
src_link = 'A-A'
tgt_dataset ='DBLP-Scholar'
tgt_link = 'A-A'



model.predict_from_pickle(src_dataset, src_link, tgt_dataset, tgt_link)
